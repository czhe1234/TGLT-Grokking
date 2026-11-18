"""Within-seed exchanges of early and late parameter groups on the original held-out grid."""
from concurrent.futures import ProcessPoolExecutor
from replay import *

STAGES=("embedding","first_block","second_block","head")


def stage(key):
    if key.startswith("blocks.0."):
        return 1
    if key.startswith("blocks.1."):
        return 2
    if key.startswith(("classifier.","final_norm.")):
        return 3
    return 0


@torch.no_grad()
def experiment(rid):
    early,late=(51000,58000) if rid.startswith("r2") else (107000,155000)
    model,_,_,_,_=load(rid,early)
    later,_,_,_,_=load(rid,late)
    initial={n:p.clone() for n,p in model.named_parameters() if p.requires_grad}
    final=dict(later.named_parameters())
    rows=[]
    for mask in range(16):
        for n,p in model.named_parameters():
            if n in initial:
                p.copy_(final[n] if mask & (1<<stage(n)) else initial[n])
        z=[]
        for chunk in I.split(256):
            h=bridge.first_layer(model,DATA["tokens"][chunk],DATA["roles"][chunk],"TGLT")
            z.append(bridge.native_parts(model,h,"TGLT")[0])
        z=torch.cat(z)
        train,test=score(z,TR),score(z)
        rows.append(dict(run_id=rid,early_step=early,late_step=late,mask=mask,
            late_groups=",".join(key for i,key in enumerate(STAGES) if mask&(1<<i)),
            train_correct=train["correct"],train_ce=train["ce"],test_correct=test["correct"],
            test_ce=test["ce"],test_margin=test["margin"]))
    return rows


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[r for per_run in pool.map(experiment,RUNS) for r in per_run]
    table(OUT/"cross_time.csv",rows)
