"""Measure local mode deletions and replay the AdamW update on private copies."""
import math
from concurrent.futures import ProcessPoolExecutor
from replay import *


def group_of(name):
    if name.startswith(("residue.", "role.", "position")):
        return "input_embedding"
    if not name.startswith("blocks."):
        return "output_norm" if name.startswith("final_norm.") else "classifier"
    _, layer, parameter, *_ = name.split(".")
    scopes = {"drive":"source","visible_factor":"interaction","mode_factor":"interaction",
              "mode_logits":"interaction","out":"write","norm":"norm"}
    return "L%d_%s" % (int(layer)+1, scopes[parameter])


def energy_by_mode(x):
    return x.square().sum(-1).mean(0)


@torch.no_grad()
def mode_metrics(model,s):
    mat = influence_matrix(model,s,7)
    rows, effects = [], []
    src = energy_by_mode(s["source"][DONOR]-s["source"])
    delivered = energy_by_mode(s["arrived"][DONOR]-s["arrived"])
    w = s["mode_w"].reshape(P,P,M,D)
    mixed = w-w.roll(-1,0)-w.roll(-1,1)+w.roll((-1,-1),(0,1))
    interaction = energy_by_mode(mixed.reshape(N,M,D))
    gates = model.blocks[0].mode_logits.softmax(0)
    for j in range(M):
        rows.append(dict(mode=j+1, source_energy=float(src[j]), received_energy=float(delivered[j]),
            received_id=operand_count(s["arrived"][:,j]), mixed_difference_energy=float(interaction[j]),
            gate=float(gates[j])))
    for mask in range(16):
        active = [(mask >> j)&1 for j in range(M)]
        delta = -(s["mode_w"]*torch.tensor(active,dtype=torch.float64)[None,:,None]).sum(1)
        z = write_output(model,s,mat,delta)
        te,tr = score(z),score(z,TR)
        effects.append(dict(mask=mask,removed=",".join(str(j+1) for j,v in enumerate(active) if v),
            test_correct=te["correct"],test_ce=te["ce"],test_margin=te["margin"],
            train_correct=tr["correct"],train_ce=tr["ce"],train_margin=tr["margin"]))
    return rows, effects


def rule_fraction(x):
    # Descriptive equal-sum variance ratio, using the known labels on the full finite grid.
    x=x-x.mean(0)
    means = x.new_zeros(P,x.shape[1]).index_add_(0,Y,x)/P
    return float(means.square().sum()*P/x.square().sum().clamp_min(1e-30))


@torch.no_grad()
def functional(model,s):
    mat = influence_matrix(model,s,7)
    z, zdel = score(s["z"]),score(write_output(model,s,mat,-s["j"]))
    return dict(test_ce=z["ce"],test_margin=z["margin"],test_correct=z["correct"],
        train_ce=score(s["z"],TR)["ce"],transport=float((s["arrived"][DONOR]-s["arrived"]).square().sum((1,2)).mean()),
        mixed_energy=float(s["j"].square().sum(1).mean()),write_rule_fraction=rule_fraction(s["w"]),
        query_rule_fraction=rule_fraction(s["q"]),joint_ce_cost=zdel["ce"]-z["ce"],
        joint_margin_cost=z["margin"]-zdel["margin"])


def check_native(model,native,s):
    ix=TE[:16]
    slow=ref.OpenLocalExponentialSweep(M,model.r,model.T,native.lambdas.tolist(),False).double()
    slow.load_state_dict(native.state_dict())
    fast=model.sweep
    model.sweep=slow
    for block in model.blocks:
        block.sweep=slow
    with torch.no_grad():
        h0=model.residue_features(DATA["tokens"][ix])+model.role_features(DATA["roles"][ix])+model.position[None]
        h1=model.blocks[0](h0)
        direct=model(DATA["tokens"][ix],DATA["roles"][ix])
        h1[:,7]=model.blocks[0].norm(s["pre7"][ix]-s["j"][ix])
        manual=model.classifier(model.final_norm(model.blocks[1](h1)[:,15]))
    model.sweep=fast
    for block in model.blocks:
        block.sweep=fast
    mat=influence_matrix(model,s,7)
    fastdel=write_output(model,s,mat,-s["j"])[ix].detach()
    e0=float((direct-s["z"][ix]).abs().max())
    e1=float((manual-fastdel).abs().max())
    assert max(e0,e1)<1e-7,(e0,e1)
    return e0,e1


def calculate(job):
    rid,step = job
    model,cp,native,path,checksum=load(rid,step)
    before=capture(model)
    direct_error,edit_error=check_native(model,native,before)
    params,grads,dirs,opt,info=training["update_vectors"](model,cp,DATA)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    originals=[p.detach().clone() for p in params]
    groups=[group_of(n) for n in names]
    old_m=[opt.state[p]["exp_avg"] for p in params]
    old_v=[opt.state[p]["exp_avg_sq"] for p in params]
    def norm(seq,indices):
        return math.sqrt(sum(float(seq[i].square().sum()) for i in indices))
    adapt=[g+m for g,m in zip(*dirs[:2])]
    full=[a+d for a,d in zip(adapt,dirs[2])]
    assert 0 < info["clip_factor"] <= 1
    by_group=[]
    for name in sorted(set(groups)) + ["all"]:
        ix=[i for i,g in enumerate(groups) if name=="all" or name==g]
        by_group.append(dict(group=name,parameter_numel=sum(params[i].numel() for i in ix),
            raw_gradient_norm=norm(grads,ix)/info["clip_factor"],first_moment_norm=norm(old_m,ix),
            second_moment_norm=norm(old_v,ix),current_gradient_update_norm=norm(dirs[0],ix),
            history_update_norm=norm(dirs[1],ix),adaptive_norm=norm(adapt,ix),
            weight_decay_norm=norm(dirs[2],ix),joint_norm=norm(full,ix)))
    with torch.no_grad():
        for i,p in enumerate(params):
            p.grad=grads[i].clone()
        opt.step()
        error=max(float((p-originals[i]-full[i]).abs().max()) for i,p in enumerate(params))
        assert error<1e-10,error
        after=capture(model)
        now=local_summary(model,after)
        mode1,delete1=mode_metrics(model,after)
        # Restore the initial downstream map before evaluating its edited activations.
        for p,v in zip(params,originals):
            p.copy_(v)
        previous=local_summary(model,before)
        base=functional(model,before)
        factors=[]
        for name in ["adaptive","decay","full"]+sorted(set(groups))+["first_layer_and_embedding","downstream"]:
            for i,p in enumerate(params):
                include=(groups[i]==name or
                         (name=="first_layer_and_embedding" and (groups[i].startswith("L1") or groups[i]=="input_embedding")) or
                         (name=="downstream" and not (groups[i].startswith("L1") or groups[i]=="input_embedding")))
                if name=="adaptive":
                    delta=adapt[i]
                elif name=="decay":
                    delta=dirs[2][i]
                elif name=="full" or include:
                    delta=full[i]
                else:
                    delta=None
                p.copy_(originals[i])
                if delta is not None:
                    p.add_(delta)
            changes=functional(model,after if name=="full" else capture(model))
            factors.append(dict(component=name,**{"delta_"+k:changes[k]-base[k] for k in changes}))
        for p,v in zip(params,originals):
            p.copy_(v)
        # Re-evaluate baseline ablations with the original downstream parameters.
        mode0,delete0=mode_metrics(model,before)
    assert digest(Path(path))==checksum
    return dict(run_id=rid,step=step,source_sha256=checksum,
        beyond_recorded_horizon=step==int(cp["config"]["steps"]),
        base=previous,after=now,modes_before=mode0,modes_after=mode1,delete_before=delete0,delete_after=delete1,
        group_norms=by_group,components=factors,update=dict(info,parameter_replay_error=error,
            native_forward_error=direct_error,native_edit_error=edit_error))


def main():
    jobs=[(rid,t) for i,rid in enumerate(RUNS) for t in ANCHORS[i//2]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        records=[]
        for r in pool.map(calculate,jobs):
            records.append(r)
            print(r["run_id"],r["step"],r["update"],flush=True)
    write_json(OUT/"updates.json",records)
    for tag in ["modes_before","modes_after","delete_before","delete_after","group_norms","components"]:
        table(OUT/(tag+".csv"),[dict(run_id=r["run_id"],step=r["step"],**row) for r in records for row in r[tag]])


if __name__=="__main__":
    main()
