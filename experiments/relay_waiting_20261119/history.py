"""Changes of numeral geometry on the original saved path and the original query computation."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"relay_learning_20261118"))
import replay as old
from order import cycles,P

OUT=HERE/"results"
POLICY=json.loads((HERE/"protocol.json").read_text())
real,shuffled=cycles("cpu",POLICY["shuffle_seed"])
runs=old.RUNS[:2]
selected=set(range(0,300001,2500))|{4000,91500,105500,107000,155000,256000,258000}
torch.set_num_threads(1)


def unit_kernel(e):
    z=e-e.mean(0)
    gram=z@z.T
    off=~torch.eye(P,dtype=torch.bool)
    k=gram[off]
    k=k-k.mean()
    return k/k.norm()


def centered_tangent(e,d):
    z=e-e.mean(0)
    dc=d-d.mean(0)
    return dc-(dc*z).sum()/z.square().sum()*z


@torch.no_grad()
def run_path(rid):
    index={}
    with (HERE.parent/"relay_learning_20261118/results/trajectories.csv").open() as f:
        for row in csv.DictReader(f):
            if row["run_id"]==rid:
                index[int(row["step"])]=row
    kernel=None
    cumulative=0.
    previous_angle=None
    rows=[]
    for t,src in sorted(index.items()):
        path=Path(src["path"])
        assert old.digest(path)==src["sha256"]
        cp=torch.load(path,map_location="cpu",weights_only=False)
        e=cp["model_state"]["residue.weight"][1:].double()
        kr=unit_kernel(e)
        rho=float(real.measure(e))
        angle=math.asin(math.sqrt(rho))
        arc=0 if kernel is None else 2*math.asin(min(1.,float((kr-kernel).norm())/2))
        gain=0 if previous_angle is None else angle-previous_angle
        assert abs(gain)<=arc+1e-10
        cumulative+=arc
        row=dict(run_id=rid,step=t,order=rho,order_random=float(shuffled.measure(e)),angle=angle,
                 kernel_arc=arc,angle_gain=gain,cumulative_arc=cumulative,
                 train_ce=float(src["original_train_ce"]),train_correct=int(src["original_train_correct"]),
                 test_correct=int(src["original_test_correct"]),test_without_query=int(src["no_query_mod_correct"]),
                 source_sha256=src["sha256"])
        if t:
            # Reconstruct a sampled previous adaptive displacement from the saved post-step AdamW moments.
            # Recover residue's optimizer ID from the original grouping and parameter name, never from its position.
            if t==500:
                m,_,_,_,_=old.load(rid,t)
                opt=old.ref.make_optimizer(m,old.ref.RunConfig(**cp["config"]))
                idx_to_param=[(p,idx) for g,h in zip(opt.param_groups,cp["optimizer_state"]["param_groups"]) for p,idx in zip(g["params"],h["params"])]
                param_id=next(i for p,i in idx_to_param if p is m.residue.weight)
            state=cp["optimizer_state"]["state"][param_id]
            group=next(g for g in cp["optimizer_state"]["param_groups"] if param_id in g["params"])
            assert state["exp_avg"].shape==cp["model_state"]["residue.weight"].shape and int(state["step"])==t
            b1,b2=group["betas"]; lr=group["lr"]
            d=-lr*(state["exp_avg"][1:].double()/(1-b1**t))/((state["exp_avg_sq"][1:].double()/(1-b2**t)).sqrt()+group["eps"])
            prior=(e-d)/(1-lr*group["weight_decay"])
            prev,g=real.measure(prior,True)
            tangent=centered_tangent(prior,d)
            row.update(sample_adaptive_norm=float(d.norm()),
                       sample_adaptive_cosine=float((tangent*g).sum()/(tangent.norm()*g.norm()).clamp_min(1e-30)),
                       sample_order_gain=rho-float(prev),
                       pure_decay_order_error=abs(float(real.measure(prior*(1-lr*group["weight_decay"]))-prev)))
        rows.append(row)
        kernel=kr
        previous_angle=angle
    assert len(rows)==601
    return rows


@torch.no_grad()
def query(job):
    rid,t=job
    m,_,_,_,_=old.load(rid,t)
    s=old.capture(m)
    bl=m.blocks[1]
    b=old.B
    v=torch.einsum("bd,mdo->bmo",s["h"][:P,15],bl.visible_factor)
    v-=v.mean(0)
    rx=s["received2"].reshape(P,P,4,8)
    aonly=rx.mean(1)
    r=(rx-rx.mean(0)).reshape(P*P,4,8)
    f=torch.einsum("nmc,mco->nmo",r,bl.mode_factor[:,:8])
    q=torch.nn.functional.linear((bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]*v[b]*f).sum(1),bl.out.weight)
    q=q-q.mean(0)
    means=torch.nn.functional.one_hot(old.Y,P).double().T@q/P
    return dict(run_id=rid,step=t,embedding_order=float(real.measure(m.residue.weight[1:])),
                local_b_order=float(real.measure(v)),received_order=float(real.measure(aonly)),
                query_sum_fraction=float(P*means.square().sum()/q.square().sum()),
                query_sum_energy=float(means.square().sum(1).mean()),query_total_energy=float(q.square().sum(1).mean()))


def main():
    with ProcessPoolExecutor(max_workers=6) as pool:
        paths=[pool.submit(run_path,r) for r in runs]
        stages=list(pool.map(query,[(r,t) for r in runs for t in sorted(selected)]))
        histories=[f.result() for f in paths]
    old.table(OUT/"original_embedding.csv",[x for rr in histories for x in rr])
    old.table(OUT/"original_query.csv",stages)
    periods=[(4000,20000),(20000,60000),(60000,90000),(20000,85000),(4000,105500),(90000,105500),(105500,155000),(200000,256000),(256000,296000)]
    rows=[]
    for rr in histories:
        by={r["step"]:r for r in rr}
        for start,end in periods:
            first,last=by[start],by[end]
            arc=last["cumulative_arc"]-first["cumulative_arc"]
            turn=last["angle"]-first["angle"]
            sel=[r for r in rr if start<r["step"]<=end]
            rows.append(dict(run_id=first["run_id"],start=start,end=end,saved_intervals=len(sel),order0=first["order"],
                order1=last["order"],arc=arc,net_angle_gain=turn,net_progress_per_arc=turn/arc,
                median_train_ce=float(torch.tensor([r["train_ce"] for r in sel],dtype=torch.double).quantile(.5)),
                positive_intervals=sum(r["angle_gain"]>0 for r in sel),
                sampled_adaptive_cosine_median=float(torch.tensor([r["sample_adaptive_cosine"] for r in sel]).quantile(.5))))
    old.table(OUT/"original_intervals.csv",rows)
    print(len(stages),sum(map(len,histories)),len(rows),flush=True)


if __name__=="__main__":
    main()
