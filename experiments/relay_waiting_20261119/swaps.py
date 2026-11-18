"""At pulse release, exchange the trained numeral table and the remaining network within the same fork."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"relay_learning_20261118"))
import replay as old
from order import cycles,P
real,random_cycle=cycles("cpu",20261119)


@torch.no_grad()
def eval_state(m):
    s=old.capture(m)
    bl=m.blocks[1]
    v=torch.einsum("bd,mdo->bmo",s["h"][:P,15],bl.visible_factor)
    v-=v.mean(0)
    rx=s["received2"].reshape(P,P,4,8)
    f=torch.einsum("nmc,mco->nmo",(rx-rx.mean(0)).reshape(P*P,4,8),bl.mode_factor[:,:8])
    q=torch.nn.functional.linear((bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]*v[old.B]*f).sum(1),bl.out.weight)
    zdel=old.at_head(m,s["pre2"]-q)
    q=q-q.mean(0)
    mu=torch.nn.functional.one_hot(old.Y,P).double().T@q/P
    tr=old.score(s["z"],old.TR)
    te=old.score(s["z"])
    return dict(train_correct=tr["correct"],test_correct=te["correct"],test_ce=te["ce"],test_margin=te["margin"],
                embedding_order=float(real.measure(m.residue.weight[1:])),query_sum_fraction=float(P*mu.square().sum()/q.square().sum()),
                query_removed_correct=old.score(zdel)["correct"])


@torch.no_grad()
def compare(job):
    fork,time=job
    cpfile="pulse_end.pt" if time==1000 else "latest.pt"
    arms=["native","shuffled","increase","decrease"]
    states={}
    for a in arms:
        cp=torch.load(HERE/".branch_state"/(fork["name"]+"__"+a)/cpfile,map_location="cpu",weights_only=False)
        assert cp["step"]==fork["step"]+time
        states[a]=cp["model_state"]
    m,_,_,_,_=old.load(fork["run"],fork["step"])
    rows=[]
    choices=[("native","native")]+[(e,r) for a in arms[1:] for e,r in [(a,"native"),("native",a),(a,a)]]
    for emb,rest in choices:
        m.load_state_dict(states[rest])
        m.sweep.rebuild()
        m.residue.weight[1:].copy_(states[emb]["residue.weight"][1:])
        metrics=eval_state(m)
        if emb==rest:
            curve=json.loads((HERE/"records"/(fork["name"]+"__"+emb)/"curve.json").read_text())
            runtime=next(x for x in curve if x["relative_step"]==time)
            assert abs(metrics["test_correct"]-runtime["test_correct"])<=2
            assert abs(metrics["embedding_order"]-runtime["embedding_order"])<1e-10
        rows.append(dict(fork=fork["name"],relative_step=time,embedding_from=emb,rest_from=rest,**metrics))
    return rows


if __name__=="__main__":
    forks=json.loads((HERE/"protocol.json").read_text())["forks"]
    with ProcessPoolExecutor(max_workers=5) as pool:
        rows=[x for rs in pool.map(compare,[(f,t) for f in forks for t in (1000,4000)]) for x in rs]
    old.table(HERE/"results/pulse_swaps.csv",rows)
    print(len(rows),flush=True)
