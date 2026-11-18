"""Donor swaps and local replacements; fit adapters only on original training examples."""
from concurrent.futures import ProcessPoolExecutor
from replay import *


def patch_row(z,base,kind,pos):
    native=base.argmax(1)
    donor=native[DONOR]
    pred=z.argmax(1)
    rows=[]
    for name,ix in (("held",TE),("both_held",TE[HELD[DONOR[TE]]])):
        diff=ix[native[ix]!=donor[ix]]
        rows.append(dict(condition=kind,position=pos,split=name,n=len(ix),
            original_prediction=int((pred[ix]==native[ix]).sum()),donor_prediction=int((pred[ix]==donor[ix]).sum()),
            original_true=int((pred[ix]==Y[ix]).sum()),donor_true=int((pred[ix]==Y[DONOR[ix]]).sum()),
            donor_natively_correct=int((donor[ix]==Y[DONOR[ix]]).sum()),different_n=len(diff),
            different_donor_prediction=int((pred[diff]==donor[diff]).sum()),
            different_original_prediction=int((pred[diff]==native[diff]).sum())))
    return rows


@torch.no_grad()
def patches(model,s):
    rows=patch_row(s["z"],s["z"],"untouched",-1)
    rows+=patch_row(s["z"][DONOR],s["z"],"whole_donor",-1)
    change={}
    for j in range(16):
        difference=s["h"][DONOR,j]-s["h"][:,j]
        if j==15:
            assert difference.abs().max()<1e-12
            delta=torch.zeros_like(s["pre2"])
        else:
            delta=change_pre(s,influence_matrix(model,s,j),difference)-s["pre2"]
        change[j]=delta
        rows+=patch_row(at_head(model,s["pre2"]+delta),s["z"],"single_site",j)
    summed=sum(change.values())
    err=float((summed-(s["pre2"][DONOR]-s["pre2"])).abs().max())
    assert err<1e-7,err
    rows+=patch_row(at_head(model,s["pre2"]+summed-change[7]),s["z"],"all_except_7",-1)
    return rows,err


def repair_stats(model,s,z,deleted):
    out=score(z)
    base=s["z"].argmax(1)[TE]
    pred=z.argmax(1)[TE]
    ab=deleted.argmax(1)[TE]
    y=Y[TE]
    centered=(z[TE]-s["z"][TE])
    centered-=centered.mean(1,keepdim=True)
    return dict(test_n=len(TE),test_correct=out["correct"],test_ce=out["ce"],test_margin=out["margin"],
        agrees_native=int((pred==base).sum()),repair_of_deletion_damage=int(((base==y)&(ab!=y)&(pred==y)).sum()),
        damaged_by_deletion=int(((base==y)&(ab!=y)).sum()),
        native_correct_lost=int(((base==y)&(pred!=y)).sum()),
        newly_correct=int(((base!=y)&(pred==y)).sum()),
        centered_logit_rmse=float(centered.square().mean().sqrt()),train_ce=score(z,TR)["ce"])


def basis(x,cap):
    x=x-x[TR].mean(0)
    _,s,v=torch.linalg.svd(x[TR],full_matrices=False)
    n=min(cap,int((s>1e-8*s[0]).sum()))
    return (x@v[:n].T)*(len(TR)**.5/s[:n])


def fit_adapter(model,s,mat,pre,del_pre,features,u=None):
    # Only frozen teacher outputs on TR supervise the replacement. No erased scalar is used as a target.
    bl=model.blocks[0]
    mat_train=mat[B[TR]].detach()
    pre_train=pre[TR].detach()
    old_h=bl.norm(pre_train).detach()
    teacher=s["z"][TR].detach()
    teacher=teacher-teacher.mean(1,keepdim=True)
    norm=teacher.square().mean(1,keepdim=True).clamp_min(1e-12)
    scale=float((pre_train-pre_train.mean(1,keepdim=True)).square().mean().sqrt())
    x=features[TR].detach()
    num_out=1 if u is not None else D
    weight=torch.zeros(x.shape[1],num_out,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([weight],max_iter=160,history_size=20,tolerance_grad=1e-11,
                          tolerance_change=1e-13,line_search_fn="strong_wolfe")
    def correction(x):
        raw=scale*(x@weight)
        return raw*u[None] if u is not None else raw
    def closure():
        opt.zero_grad()
        nh=bl.norm(pre_train+correction(x))
        shift=torch.bmm((nh-old_h)[:,None],mat_train).squeeze(1)
        logits=at_head(model,del_pre[TR]+shift)
        logits=logits-logits.mean(1,keepdim=True)
        loss=((logits-teacher).square()/norm).mean()+1e-8*weight.square().mean()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        delta=correction(features)
        dh=bl.norm(pre+delta)-bl.norm(pre)
        change=torch.einsum("abk,bkd->abd",dh.reshape(P,P,D),mat).reshape(N,D)
        logits=at_head(model,del_pre+change)
        centered=logits[TR]-logits[TR].mean(1,keepdim=True)
        loss=float(((centered-teacher).square()/norm).mean())
        return logits,dict(n_features=x.shape[1],n_outputs=num_out,fit_iterations=opt.state[weight]["n_iter"],
                           train_relative_logit_mse=loss),delta


def pre8_and_right(model):
    rows=[]
    for start in range(0,N,192):
        t=DATA["tokens"][start:start+192]
        r=DATA["roles"][start:start+192]
        h=model.residue_features(t)+model.role_features(r)+model.position[None]
        bl=model.blocks[0]
        drive=bl.drive(h).reshape(len(t),16,M,8)
        right,left=bl.sweep(drive)
        v=torch.einsum("nd,mdo->nmo",h[:,8],bl.visible_factor)
        a=torch.einsum("nmc,mco->nmo",torch.cat([right[:,8],left[:,8]],-1),bl.mode_factor)
        x=bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]*v*a
        rows.append(h[:,8]+bl.out(x.sum(1)))
    return torch.cat(rows)


def repairs(model,s):
    model.requires_grad_(False)
    mat=influence_matrix(model,s,7)
    bl=model.blocks[0]
    U,S,_=torch.linalg.svd(bl.out.weight,full_matrices=False)
    u=U[:,8]
    erased=(s["w"]@u)[:,None]*u
    pre=s["pre7"]-erased
    del_pre=change_pre(s,mat,bl.norm(pre)-s["h"][:,7])
    deleted=at_head(model,del_pre)
    rows=[]
    def add(name,z,**attrs):
        rows.append(dict(condition=name,**repair_stats(model,s,z,deleted),**attrs))
    add("untouched",s["z"])
    add("delete_u9",deleted)
    restore=write_output(model,s,mat,-erased+erased)
    assert (restore-s["z"]).abs().max()<1e-8
    add("exact_restore_check",restore)
    for j in (7,9):
        v=U[:,j]
        add("delete_u%d"%(j+1),write_output(model,s,mat,-(s["w"]@v)[:,None]*v))
    gen=torch.Generator().manual_seed(1701)
    for j in range(5):
        v=torch.randn(D,dtype=torch.float64,generator=gen)
        v=v-v.dot(u)*u
        v=v/v.norm()
        add("matched_perturbation_%d"%j,write_output(model,s,mat,-(s["w"]@u)[:,None]*v))
    # Donor swaps isolate u9 and the remaining write, keeping b and the residual stream fixed.
    donor_w=s["w"][DONOR]-s["w"]
    donor_u=(donor_w@u)[:,None]*u
    patch=[]
    for name,change in (("swap_u9",donor_u),("swap_rest_write",donor_w-donor_u),("swap_all_write",donor_w)):
        z=write_output(model,s,mat,change)
        patch+=patch_row(z,s["z"],name,7)
    r=basis(s["arrived"][:,3],8)
    local=basis(s["local"],58)
    ones=torch.ones(N,1,dtype=torch.float64)
    additive=torch.cat([ones,r,local],1)
    product=torch.cat([additive,(r[:,:,None]*local[:,None,:]).flatten(1)],1)
    surviving=s["w"]-erased
    # Remove roundoff along u before the PCA; cap the retained dimensions.
    surviving=surviving-(surviving@u)[:,None]*u
    orth=torch.cat([ones,basis(surviving,32)],1)
    pre8=pre8_and_right(model)
    assert (bl.norm(pre8)-s["h"][:,8]).abs().max()<1e-9
    site8=torch.cat([ones,basis(s["h"][:,8],32)],1)
    setups=(("constant",ones,7),("operand_b_only",torch.cat([ones,local],1),7),
            ("received4_only",torch.cat([ones,r],1),7),("received4_plus_b",additive,7),
            ("received4_times_b",product,7),("surviving_write",orth,7),("site8_linear",site8,8))
    sample=TE[:24]
    physical_errors=[]
    for name,x,pos in setups:
        z,info,delta=fit_adapter(model,s,mat if pos==7 else influence_matrix(model,s,8),
                                 pre if pos==7 else pre8,del_pre,x,u if pos==7 else None)
        # Independent check through the full original downstream block, with true LayerNorm.
        h=s["h"][sample].clone()
        h[:,7]=bl.norm(pre[sample])
        h[:,pos]=bl.norm((pre if pos==7 else pre8)[sample]+delta[sample])
        actual=model.classifier(model.final_norm(model.blocks[1](h)[:,15]))
        error=float((actual-z[sample]).abs().max())
        assert error<1e-7,error
        physical_errors.append(error)
        add(name,z,**info)
        if name=="received4_times_b":
            # Shuffle the completed correction across unseen examples without changing its values.
            perm=TE[torch.randperm(len(TE),generator=gen)]
            shuffled=delta.clone()
            shuffled[TE]=delta[perm]
            add("shuffled_local_repair",write_output(model,s,mat,-erased+shuffled))
    # Harder replacement: erase every first-layer write at site 7; retain its b-only residual.
    pre_blank7=s["local"]+bl.out.bias
    blank_pre=change_pre(s,mat,bl.norm(pre_blank7)-s["h"][:,7])
    blank=at_head(model,blank_pre)
    rows.append(dict(condition="remove_whole_write7",**repair_stats(model,s,blank,blank)))
    x8=site8[:,1:9]
    i,j=torch.triu_indices(8,8)
    quadratic=torch.cat([site8,x8[:,i]*x8[:,j]],1)
    for name,x in (("whole7_repair_at8_linear",site8),("whole7_repair_at8_quadratic",quadratic)):
        z,info,delta=fit_adapter(model,s,influence_matrix(model,s,8),pre8,blank_pre,x)
        h=s["h"][sample].clone()
        h[:,7]=bl.norm(pre_blank7[sample])
        h[:,8]=bl.norm(pre8[sample]+delta[sample])
        native_z=model.classifier(model.final_norm(model.blocks[1](h)[:,15]))
        physical_errors.append(float((z[sample]-native_z).abs().max()))
        assert physical_errors[-1]<1e-7
        rows.append(dict(condition=name,**repair_stats(model,s,z,blank),**info))
    coeff=(s["w"]@u).reshape(P,P)
    total=coeff.var(unbiased=False)
    var_a=float(coeff.mean(1).var(unbiased=False)/total)
    var_b=float(coeff.mean(0).var(unbiased=False)/total)
    return rows,patch,dict(u9_variance_fraction_a=var_a,u9_variance_fraction_b=var_b,
                           u9_variance_fraction_joint=1-var_a-var_b, singular_value_9=float(S[8]),native_restore_error=float((restore-s["z"]).abs().max()),
                           max_repair_forward_error=max(physical_errors))


def compute(job):
    rid,step=job
    model,_,native,path,checksum=load(rid,step)
    model.requires_grad_(False)
    s=capture(model)
    rows,err=patches(model,s)
    repairs_out,checks=[],{}
    if rid.startswith("r2") and step==100000:
        repairs_out,extra,checks=repairs(model,s)
        rows+=extra
    assert digest(Path(path))==checksum
    print(rid,step,"finished",flush=True)
    return dict(run_id=rid,step=step,patches=rows,repairs=repairs_out,checks=dict(sum_sites_error=err,**checks))


def main():
    jobs=[(rid,t) for i,rid in enumerate(RUNS) for t in ANCHORS[i//2]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(compute,jobs))
    write_json(OUT/"relay.json",records)
    for name in ("patches","repairs"):
        table(OUT/(name+".csv"),[dict(run_id=r["run_id"],step=r["step"],**row) for r in records for row in r[name]])


if __name__=="__main__":
    main()
