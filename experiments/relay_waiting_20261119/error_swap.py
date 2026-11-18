"""Cross saved-state training residuals on common input/output coordinates for one actual AdamW update."""
import argparse
from copy import deepcopy
from branch import HERE,POLICY,build,cycles,data,dev,evaluate,get,ref,save,torch,F

PAIRS=[(40000,40000),(85000,85000),(105500,105500),(107000,107000),(155000,155000),(105500,256000)]
NATIVE={"b0.25_wd0.03_s0":"GPU-70b6fceb-c7c4-9467-3c81-097c9284c160",
        "b0.25_wd0.03_s2":"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"}
CACHE=HERE/".branch_state"/"residuals"
CACHE.mkdir(exist_ok=True)

def residual_name(r,t):
    return CACHE/(r+"_%07d.pt"%t)


def run(rid,stage):
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"]==NATIVE[rid]
    names=list(NATIVE)
    me=names.index(rid)
    other=names[1-me]
    order,shuffle=cycles(dev,POLICY["shuffle_seed"])
    result=[]
    for pair in PAIRS:
        t,ot=pair[me],pair[1-me]
        cp,_,_=get(rid,t)
        m,opt,cfg,_=build(cp)
        sets,tr,te,y=data(rid,cfg)
        tx,roles,ys=sets[0]
        ck=residual_name(rid,t)
        if stage=="extract":
            m.train()
            logits=m(tx,roles)
            error=torch.autograd.grad(F.cross_entropy(logits,ys),logits)[0].detach().cpu()
            if ck.exists():
                assert torch.equal(torch.load(ck,map_location="cpu",weights_only=True),error)
            else:
                torch.save(error,ck)
            continue
        base=evaluate(m,sets,tr,te,y,order,shuffle)
        es=torch.load(ck,map_location="cpu",weights_only=True).to(dev)
        ed=torch.load(residual_name(other,ot),map_location="cpu",weights_only=True).to(dev)
        own_norm=float(es.double().norm()); donor_norm=float(ed.double().norm())
        assert own_norm>0 and donor_norm>0
        for arm in ("native","own_residual","zero_current","donor","donor_norm_matched"):
            m,opt,cfg,_=build(deepcopy(cp))
            m.train()
            for group in opt.param_groups:
                group["lr"]=ref.learning_rate(t+1,cfg)
            opt.zero_grad(set_to_none=True)
            logits=m(tx,roles)
            if arm=="native":
                F.cross_entropy(logits,ys).backward()
            else:
                error={"own_residual":es,"zero_current":torch.zeros_like(es),
                       "donor":ed,"donor_norm_matched":ed*(own_norm/donor_norm)}[arm]
                logits.backward(error)
            norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip,error_if_nonfinite=True))
            opt.step()
            if arm=="native":
                after_parameters={k:p.detach().clone() for k,p in m.named_parameters()}
            elif arm=="own_residual":
                assert all(torch.equal(after_parameters[k],p) for k,p in m.named_parameters())
            after=evaluate(m,sets,tr,te,y,order,shuffle)
            result.append(dict(recipient=rid,recipient_step=t,donor=other,donor_step=ot,arm=arm,
                               receiver_residual_norm=own_norm,donor_residual_norm=donor_norm,
                               nonzero_error_examples=int((es.double().norm(dim=1)>0).sum()),
                               parameter_gradient_norm=norm,
                               **{"d_"+k:after[k]-base[k] for k in base}))
    if stage!="extract":
        save(HERE/"results"/("error_swap_"+rid+".json"),result)
        print(rid,len(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("rid")
    p.add_argument("stage",choices=["extract","apply"])
    args=p.parse_args()
    run(args.rid,args.stage)
