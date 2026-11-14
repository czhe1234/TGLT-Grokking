"""Check completed training, saved states, common metrics and SwanLab source equality."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:os.environ[k]="1"
from pathlib import Path
import sys,json,math,hashlib,sqlite3,urllib.request,urllib.parse
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B));import analyze as a
import torch
torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
d=a.make_data();reports=[];hashes=[]
for row in a.transformer_roster():
    dest=Path(row["path"]);status=json.loads((dest/"status.json").read_text())
    assert status["state"]=="complete" and status["step"]==100000
    history=json.loads((dest/"history.json").read_text())
    assert [r["step"] for r in history]==list(range(0,100001,500))
    paths=sorted(dest.glob("checkpoint_*.pt"))
    assert [int(p.stem.split("_")[-1]) for p in paths]==list(range(0,100001,500))
    for r in history:
        assert r["train_n"]==870 and r["test_n"]==2611
        for pre in ["train","test"]:
            assert abs(r[pre+"_acc"]-r[pre+"_correct"]/r[pre+"_n"])<1e-15
        assert all(math.isfinite(v) for v in r.values())
    cfg=json.loads((dest/"config.json").read_text());origin=json.loads((dest/"origin.json").read_text())
    assert origin["gpu_uuid"] in a.ALLOWED
    for name,digest in origin["code_hashes"].items():assert a.sha(B/name)==digest,(row["run_id"],name)
    m,cp,_=a.load_model(row,100000)
    assert cp["step"]==100000 and cp["config"]==cfg
    assert torch.equal(cp["train_idx"],d["train"]) and torch.equal(cp["test_idx"],d["test"])
    assert cp["data_sha256"]==a.sha(B/"data.pt")
    states=cp["optimizer_state"]["state"]
    assert states and all(int(x["step"])==100000 for x in states.values())
    assert set(cp["rng_state"])=={"python","torch","cuda"}
    assert len(cp["rng_state"]["cuda"])==1
    with torch.no_grad():
        z=torch.cat([m(d["tokens"][i:i+256],d["roles"][i:i+256]) for i in range(0,3481,256)])
    correct=z.argmax(-1)==d["labels"]
    replay={k:int(correct[d[k]].sum()) for k in ["train","test"]}
    dif={k:replay[k]-history[-1][k+"_correct"] for k in replay}
    assert max(abs(x) for x in dif.values())<=1, (row["run_id"],dif)
    for p in paths:hashes.append(dict(run_id=row["run_id"],step=int(p.stem.split("_")[-1]),sha256=a.sha(p),bytes=p.stat().st_size,path=str(p)))
    reports.append(dict(run_id=row["run_id"],saved_points=len(paths),gpu_uuid=origin["gpu_uuid"],
        final_train=history[-1]["train_acc"],final_test=history[-1]["test_acc"],
        CPU_float64_replay_count_difference=dif,optimizer_step=100000))
    print(json.dumps(reports[-1]),flush=True)
a.write_csv(hashes,B/"validation/checkpoint_manifest.csv")
# Initial parameter equality for each seed across window masks, including all-vs-none.
init_hashes={}
for row in a.transformer_roster():
    cp=torch.load(Path(row["path"])/"checkpoint_0000000.pt",map_location="cpu",weights_only=False)
    m=a.WindowTransformer(row["seed"],row["window"]);m.load_state_dict(cp["model_state"])
    init_hashes.setdefault(row["seed"],[]).append(a.param_hash(m))
assert all(len(set(v))==1 for v in init_hashes.values())
# W=15 and unmasked attention expose identical token pairs.
global_equivalence=[]
for seed in range(3):
    paths=[B/"runs"/("%s_s%d"%(prefix,seed)) for prefix in ["transformer_W15","transformer_global"]]
    hs=[json.loads((p/"history.json").read_text()) for p in paths]
    metric_keys=["train_acc","test_acc","train_ce","test_ce"]
    diffs={key:max(abs(x[key]-y[key]) for x,y in zip(*hs)) for key in metric_keys}
    cps=[torch.load(p/"checkpoint_0100000.pt",map_location="cpu",weights_only=False)["model_state"] for p in paths]
    max_diff=max(float((cps[0][k]-cps[1][k]).abs().max()) for k in cps[0])
    global_equivalence.append(dict(seed=seed,max_history_differences=diffs,max_final_state_difference=max_diff))
# Mechanism data and additive partition checks.
mechan=[]
for row in a.baseline_roster()+a.transformer_roster():
    if row["window"] not in [8,12]:continue
    dest=B/"mechanism"/row["run_id"]
    m=json.loads((dest/"complete.json").read_text())
    assert m["points"]==(201 if row["kind"]=="Transformer" else 101)
    assert m["max_partition_error"]<1e-8
    mechan.append(m)
# Check dashboard scalar values against original histories, in their displayed units.
db=sqlite3.connect(str(B/"swanlab/logs/runs.swanlab"))
exps=db.execute("select id,name from experiment").fetchall()
assert len(exps)==15
mapping={"准确率/训练集（%）":("train_acc",100),"准确率/测试集（%）":("test_acc",100),
"损失/训练交叉熵":("train_ce",1),"损失/测试交叉熵":("test_ce",1),
"分类间隔/测试集平均":("test_margin_mean",1),"参数/参数范数":("weight_norm",1)}
names={}
for row in a.transformer_roster():
    name=("全局" if row["window"] is None else "窗口%d"%row["window"])+"_seed%d"%row["seed"]
    names[name]=row
count=0
for eid,name in exps:
    row=names[name];hist=json.loads((Path(row["path"])/"history.json").read_text())
    receipt=json.loads((Path(row["path"])/"swanlab.json").read_text())
    assert receipt["finished"] and receipt["last_step"]==100000
    for key,(source,factor) in mapping.items():
        url="http://127.0.0.1:16097/api/v1/experiment/%d/tag/%s"%(eid,urllib.parse.quote(key,safe=""))
        data=json.load(urllib.request.urlopen(url))["data"]["list"]
        got={int(v["index"]):float(v["data"]) for v in data}
        expected={int(r["step"]):r[source]*factor for r in hist}
        assert got.keys()==expected.keys(),(name,key,len(got))
        assert all(math.isclose(got[s],v,rel_tol=1e-6,abs_tol=1e-7) for s,v in expected.items()),(name,key)
        count+=len(got)
result=dict(passed=True,training_runs=15,checkpoints=len(hashes),history_points=15*201,
    final_checks=reports,initialization_matched=True,global_equivalence=global_equivalence,mechanism_runs=len(mechan),
    mechanism_points=sum(z["points"] for z in mechan),max_partition_error=max(z["max_partition_error"] for z in mechan),
    swanlab_runs=15,swanlab_scalars_verified=count)
a.json_write(result,B/"validation/completion_checks.json")
print(json.dumps({k:v for k,v in result.items() if k!="final_checks"}),flush=True)
