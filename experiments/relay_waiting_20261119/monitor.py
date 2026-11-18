"""Record completed forks as they finish and verify the local SwanLab UI against native logs."""
import json
import math
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="",SWANLAB_MODE="local")
HERE=Path(__file__).resolve().parent
PROTO=json.loads((HERE/"protocol.json").read_text())
LOG=HERE/"swanlab/logs"
R=HERE/"swanlab/receipt.json"
PORT=16104
NAMES=[f["name"]+"__"+arm for f in PROTO["forks"] for arm in PROTO["arms"]]
M={"表现/训练准确率":("train_correct",100/870),"表现/留出准确率":("test_correct",100/2611),
   "表现/训练交叉熵":("train_ce",1),"表现/留出交叉熵":("test_ce",1),"表现/留出分类间隔":("test_margin",1),
   "因果干预/去掉查询调制的留出准确率":("query_removed_correct",100/2611),
   "数的排列/数字编码":("embedding_order",100),"数的排列/随机顺序":("shuffled_order",100),
   "数的排列/到达查询的第一个数":("received_order",100),"数的排列/查询的第二个数":("local_b_order",100),
   "查询交互/同和分量百分比":("query_sum_fraction",100)}
def vals(row):
    out={k:float(row[c])*v for k,(c,v) in M.items()}
    out["因果干预/去掉查询调制损失的百分点"]=100*(row["test_correct"]-row["query_removed_correct"])/2611
    return out
def records(n):
    return json.loads((HERE/"records"/n/"curve.json").read_text())
def sync():
    import swanlab
    LOG.mkdir(parents=True,exist_ok=True)
    already=json.loads(R.read_text())["done"] if R.exists() else []
    while len(already)<len(NAMES):
        for name in NAMES:
            d=HERE/"records"/name
            if name in already or not (d/"complete.json").exists(): continue
            meta=json.loads((d/"meta.json").read_text())
            swanlab.init(project="TGLT-Waiting-Cyclic-Order",experiment_name=name,group=name.split("__")[0],
                         mode="local",log_dir=str(LOG),config=dict(meta,protocol=PROTO,train_size=870,test_size=2611),
                         settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
            for r in records(name):
                swanlab.log(vals(r),step=r["relative_step"])
            swanlab.finish()
            already.append(name)
            tmp=R.with_suffix(".tmp");tmp.write_text(json.dumps(dict(done=already,version=swanlab.__version__)));tmp.replace(R)
        if len(already)<len(NAMES): time.sleep(5)
def verify():
    with sqlite3.connect(LOG/"runs.swanlab") as db:
        runs=db.execute("select id,name from experiment").fetchall()
    assert set(n for _,n in runs)==set(NAMES) and len(runs)==12
    count=0
    for ix,name in runs:
        rv={r["relative_step"]:vals(r) for r in records(name)}
        assert len(rv)==41
        for metric in next(iter(rv.values())):
            req="http://127.0.0.1:%d/api/v1/experiment/%d/tag/%s"%(PORT,ix,urllib.parse.quote(metric,safe=""))
            response=json.load(urllib.request.urlopen(req))
            actual={int(r["index"]):float(r["data"]) for r in response["data"]["list"]}
            assert actual.keys()==rv.keys() and all(math.isclose(actual[t],v[metric],abs_tol=1e-8,rel_tol=1e-6) for t,v in rv.items()),(name,metric)
            count+=len(rv)
    v=dict(runs=len(runs),observations=12*41,verified_scalars=count,local_port=PORT,cloud_upload=False)
    (HERE/"results/swanlab_validation.json").write_text(json.dumps(v,indent=2))
    print(v)
if __name__=="__main__":
    verify() if "--verify" in sys.argv else sync()
