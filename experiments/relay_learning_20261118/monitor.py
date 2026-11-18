"""Mirror the mechanism trajectories and checkpoint interventions in the local SwanLab viewer."""
import csv
import json
import math
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="",SWANLAB_MODE="local")
HERE=Path(__file__).resolve().parent
RES=HERE/"results"
LOG=HERE/"swanlab/logs"
PORT=16103
RUNS=("b0.25_wd0.03_s0","b0.25_wd0.03_s2","r2_T4_M4_seed1","r2_T4_M4_seed0")
NAMES=("H12_seed0","H12_seed2","H8_seed1","H8_seed0")


def read_rows(name):
    with (RES/name).open() as handle:
        return list(csv.DictReader(handle))


def values():
    out={}
    for r in read_rows("trajectories.csv"):
        point={}
        for name,col in (("表现/留出题", "test_correct"),("识别/到达7的第一个数","received_id"),
                         ("读出/位置7写回","local_write_rule"),("读出/查询收到的信息","query_input_rule"),
                         ("读出/查询交互后","query_pre_rule"),("干预/移除位置7联合项","joint_removed_correct"),
                         ("干预/移除查询位置调制","no_query_mod_correct"),("干预/仅从位置7传递第一个数","only7_correct"),
                         ("干预/移除位置7的第一个数","no7_correct")):
            point[name]=100*int(r[col])/2611
        point["表现/训练交叉熵"]=float(r["train_ce"])
        out[r["run_id"],int(r["step"])]=point
    for r in json.loads((RES/"updates.json").read_text()):
        v=out[r["run_id"],r["step"]]
        v["更新/原始梯度范数"]=r["update"]["gradient_norm"]
        for c in r["components"]:
            if c["component"] in ("full","adaptive","decay"):
                v["更新/留出间隔变化_"+c["component"]]=c["delta_test_margin"]
    for r in read_rows("patches.csv"):
        if r["split"]=="both_held" and r["condition"]=="single_site" and r["position"]=="7":
            v=out[r["run_id"],int(r["step"])]
            for name,col in (("跟随参照预测","donor_prediction"),("算对参照题","donor_true")):
                v["位置7替换/"+name]=100*int(r[col])/int(r["n"])
    for r in read_rows("repairs.csv"):
        if r["condition"] in ("delete_u9","operand_b_only","surviving_write","site8_linear","whole7_repair_at8_quadratic"):
            out[r["run_id"],int(r["step"])]["恢复/"+r["condition"]]=100*int(r["test_correct"])/2611
    return out


def native_values():
    out={}
    for r in json.loads((RES/"native_effects.json").read_text()):
        base,after=r["native_before"]["test"],r["native_after"]["test"]
        v={"原设备/裁剪前梯度范数":r["raw_grad_norm"],"原设备/实际间隔变化":after["margin"]-base["margin"],
           "原设备/留出答对数变化":after["correct"]-base["correct"],"原设备/训练交叉熵":r["native_before"]["train"]["ce"]}
        for c in r["components"]:
            if c["component"] in ("adaptive","decay","first_layer_and_embedding","downstream"):
                v["固定位移干预/"+c["component"]]=c["delta_test_margin"]
        out[r["run_id"],r["step"]]=v
    return out


def main():
    stage="--native" in sys.argv
    data=native_values() if stage else values()
    receipt=HERE/"swanlab"/("native_receipt.json" if stage else "receipt.json")
    if "--verify" in sys.argv:
        with sqlite3.connect(LOG/"runs.swanlab") as db:
            gids=db.execute("select id,name from experiment").fetchall()
        assert set(n for _,n in gids)==set(NAMES)|{"FP32_"+n for n in NAMES}
        ndata=native_values()
        count=0
        for eid,name in gids:
            native=name.startswith("FP32_")
            rid=RUNS[NAMES.index(name[5:] if native else name)]
            source=ndata if native else data
            points={s:v for (r,s),v in source.items() if r==rid}
            for key in set().union(*(v.keys() for v in points.values())):
                expected={s:v[key] for s,v in points.items() if key in v}
                url="http://127.0.0.1:%d/api/v1/experiment/%d/tag/%s"%(PORT,eid,urllib.parse.quote(key,safe=""))
                resp=json.load(urllib.request.urlopen(url))
                actual={int(z["index"]):float(z["data"]) for z in resp["data"]["list"]}
                assert actual.keys()==expected.keys(),(rid,key)
                assert all(math.isclose(actual[s],v,rel_tol=1e-6,abs_tol=1e-8) for s,v in expected.items()),(rid,key)
                count+=len(actual)
        assert count==sum(map(len,data.values()))+sum(map(len,ndata.values()))
        d=dict(runs=len(gids),checkpoints=len(data),native_updates=len(ndata),scalars=count,port=PORT,mode="local",verified=True)
        (RES/"swanlab_validation.json").write_text(json.dumps(d,indent=2))
        print(d)
        return
    if receipt.exists():
        assert json.loads(receipt.read_text())["complete"]==list(RUNS)
        return
    import swanlab
    done=[]
    for rid,name in zip(RUNS,NAMES):
        swanlab.init(project="TGLT-Relay-Learning",experiment_name=("FP32_"+name if stage else name),mode="local",log_dir=str(LOG),
            config=dict(run_id=rid,source="original checkpoints",train_n=870,test_n=2611,
                        split_seed=0,metric_units="accuracies in percent; margins in raw logits"),
            settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
        for (run,step),v in sorted(data.items()):
            if run==rid:
                swanlab.log(v,step=step)
        swanlab.finish()
        done.append(rid)
        receipt.write_text(json.dumps(dict(complete=done,version=swanlab.__version__),indent=2))


if __name__=="__main__":
    main()
