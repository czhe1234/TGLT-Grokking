"""Apply the actual historical-precision parameter displacement to stage diagnostics."""
from concurrent.futures import ProcessPoolExecutor
from updates import *


@torch.no_grad()
def one(record):
    rid,step=record["run_id"],record["step"]
    model,cp,_,_,_=load(rid,step)
    path=HERE/".native_cache"/(rid+"_%07d.pt"%step)
    replayed=torch.load(path,map_location="cpu",weights_only=False)
    before=capture(model)
    baseline=functional(model,before)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    start={n:p.clone() for n,p in params.items()}
    decay={n:replayed["decayed"][n].double()-p for n,p in start.items()}
    adap={n:replayed["after"][n].double()-replayed["decayed"][n].double() for n in start}
    full={n:replayed["after"][n].double()-p for n,p in start.items()}
    assert all(torch.equal(decay[n]+adap[n],full[n]) for n in params)
    groups=sorted(set(map(group_of,params)))
    group_values=[]
    for scope in groups+["all"]:
        names=[n for n in params if scope=="all" or group_of(n)==scope]
        def norm(v):
            return float(sum(v[n].double().square().sum() for n in names).sqrt())
        group_values.append(dict(group=scope,adaptive_norm=norm(adap),weight_decay_norm=norm(decay),
            full_norm=norm(full),clipped_gradient_norm=norm(replayed["grad"])))
    changes=[]
    for name in ["adaptive","decay","full"]+groups+["first_layer_and_embedding","downstream"]:
        for n,p in params.items():
            scope=group_of(n)
            front=scope.startswith("L1_") or scope=="input_embedding"
            p.copy_(start[n])
            if name=="adaptive":
                p.add_(adap[n])
            elif name=="decay":
                p.add_(decay[n])
            elif name=="full" or name==scope or (name=="first_layer_and_embedding" and front) or (name=="downstream" and not front):
                p.add_(full[n])
        s=capture(model)
        val=functional(model,s)
        changes.append(dict(component=name,**{"delta_"+k:val[k]-baseline[k] for k in val}))
        if name=="full":
            assert all(torch.equal(p,replayed["after"][n].double()) for n,p in params.items())
            after=local_summary(model,s)
            modes,effects=mode_metrics(model,s)
    return dict(run_id=rid,step=step,device=record["device"],raw_grad_norm=record["gradnorm"],
        clip_factor=record["clip_factor"],parameter_decomposition_exact=True,
        native_before=record["before"],native_after=record["after"],base_function=baseline,
        after_stages=after,mode_after=modes,deletion_after=effects,groups=group_values,components=changes)


if __name__=="__main__":
    data=[]
    for chip in ("A800","H100"):
        data+=read_json(OUT/("native_"+chip+".json"))
    with ProcessPoolExecutor(max_workers=5) as pool:
        records=list(pool.map(one,[r for r in data if r["step"]]))
    write_json(OUT/"native_effects.json",records)
    table(OUT/"native_components.csv",[dict(run_id=r["run_id"],step=r["step"],**v) for r in records for v in r["components"]])
