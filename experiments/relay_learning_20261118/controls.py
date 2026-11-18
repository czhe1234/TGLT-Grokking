"""Normalize edited signals at matched amplitude and verify the native float32 AdamW replay."""
from concurrent.futures import ProcessPoolExecutor
from replay import *


def matched(candidate, original):
    # LayerNorm removes the constant feature component; match the centered norm per input.
    candidate=candidate-candidate.mean(1,keepdim=True)
    ref=original-original.mean(1,keepdim=True)
    return candidate*(ref.norm(dim=1,keepdim=True)/candidate.norm(dim=1,keepdim=True).clamp_min(1e-20))


@torch.no_grad()
def interventions(model,s):
    b=model.blocks[1]
    vis=torch.einsum("bd,mdo->bmo",s["h"][:P,15],b.visible_factor)
    vis=vis-vis.mean(0)
    rx=s["received2"]-s["received2"].reshape(P,P,M,8).mean(0)[B]
    x=torch.einsum("nmc,mco->nmo",rx,b.mode_factor[:,:8])*b.local_coupling*b.mode_logits.softmax(0)[None,:,None]
    by_mode=F.linear(x*vis[B],b.out.weight)
    qmod=by_mode.sum(1)
    mat=influence_matrix(model,s,7)
    results=[]
    def add(name,z):
        results.append(dict(condition=name,**{"test_"+k:v for k,v in score(z).items()}))
    add("native",s["z"])
    for mask in range(1,16):
        ix=[j for j in range(M) if mask&(1<<j)]
        add("remove_query_mod_%02d"%mask,at_head(model,s["pre2"]-by_mode[:,ix].sum(1)))
    for shift in (1,17,31):
        wrong=F.linear(x*vis[(B+shift)%P],b.out.weight).sum(1)
        add("query_mismatched_b_%d"%shift,at_head(model,s["pre2"]-qmod+matched(wrong,qmod)))
        donor=((A+shift)%P)*P+B
        add("site7_mismatched_a_%d"%shift,write_output(model,s,mat,-s["j"]+matched(s["j"][donor],s["j"])))
    return results


def f32model(rid,step):
    model,cp,native,_,_=load(rid,step)
    model.sweep=native
    for b in model.blocks:
        b.sweep=native
    model.float()
    opt=ref.make_optimizer(model,ref.RunConfig(**cp["config"]))
    opt.load_state_dict(cp["optimizer_state"])
    for g in opt.param_groups:
        g["lr"]=ref.learning_rate(step+1,ref.RunConfig(**cp["config"]))
    with torch.no_grad():
        def full_scores():
            z=torch.cat([model(DATA["tokens"][ix],DATA["roles"][ix]) for ix in I.split(192)])
            return dict(train=score(z,TR),test=score(z))
        base=full_scores()
    opt.zero_grad(set_to_none=True)
    z=model(DATA["tokens"][TR],DATA["roles"][TR])
    F.cross_entropy(z,Y[TR]).backward()
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cp["config"]["grad_clip"],error_if_nonfinite=True))
    opt.step()
    with torch.no_grad():
        after=full_scores()
    return dict(raw_grad_norm=grad,base=base,after=after)


def run(job):
    rid,step=job
    model,cp,_,_,_=load(rid,step)
    counterfactuals=interventions(model,capture(model))
    fp32=f32model(rid,step)
    return dict(run_id=rid,step=step,counterfactuals=counterfactuals,original_float32=fp32)


if __name__=="__main__":
    jobs=[(rid,k) for i,rid in enumerate(RUNS) for k in ANCHORS[i//2]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,jobs))
    write_json(OUT/"controls.json",rows)
    table(OUT/"matched_controls.csv",[dict(run_id=r["run_id"],step=r["step"],**c) for r in rows for c in r["counterfactuals"]])
