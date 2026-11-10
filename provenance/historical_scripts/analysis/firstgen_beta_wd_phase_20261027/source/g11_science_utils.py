import os,sys,json,hashlib,inspect,importlib.util,random,copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
SOURCE=ROOT/"source"/"exp2_sequence_local_grokking.py"
EXPECTED="e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73"
ALLOWED=("GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009","GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0","GPU-70b6fceb-c7c4-9467-3c81-097c9284c160")
WEAK=8.384404281976212e-13
MOD=None
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w") as f:
        json.dump(clean(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def native():
    global MOD
    if MOD is None:
        assert sha(SOURCE)==EXPECTED
        spec=importlib.util.spec_from_file_location("clean11_historical",SOURCE)
        MOD=importlib.util.module_from_spec(spec);sys.modules[spec.name]=MOD;spec.loader.exec_module(MOD)
    return MOD
def model_from_config(cfg,device="cpu",dtype=torch.float64):
    mod=native()
    kw={k:v for k,v in cfg.items() if k in inspect.signature(mod.SequenceLocalModeModel).parameters}
    kw["lambdas"]=cfg.get("lambda_values") or mod.xi_spread_lambdas(cfg["M"],cfg["xi_min"],cfg["xi_max"]).tolist()
    return mod.SequenceLocalModeModel(**kw).to(device=device,dtype=dtype)
def data(cfg,device="cpu"):
    mod=native()
    tokens,roles,labels,_=mod.build_sequence_data(cfg["p"],cfg["task"],cfg["seq_len"],cfg["pos_a"],cfg["pos_b"],cfg["query_pos"],False,cfg["random_label_seed"])
    tr,te=mod.make_split(labels.numel(),cfg["train_fraction"],cfg["split_seed"])
    sp=json.loads((PROJECT/"week/v1/results"/Path(cfg["out_dir"]).name/"split.json").read_text())
    assert torch.equal(tr,torch.tensor(sp["train_idx"])) and torch.equal(te,torch.tensor(sp["test_idx"]))
    return tokens.to(device),roles.to(device),labels.to(device),tr.to(device),te.to(device)
def cpu_copy(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return x
def exact(a,b,path="root"):
    if torch.is_tensor(b):assert torch.equal(a.detach().cpu(),b.detach().cpu()),path
    elif isinstance(b,dict):
        assert set(a)==set(b),path
        for k in b:exact(a[k],b[k],path+"."+str(k))
    elif isinstance(b,(list,tuple)):
        assert len(a)==len(b),path
        for i,(u,v) in enumerate(zip(a,b)):exact(u,v,path+"."+str(i))
    elif isinstance(b,np.ndarray):assert np.array_equal(a,b),path
    else:assert a==b,path
def restore(model,opt,cp):
    model.load_state_dict(cp["model_state"],strict=True)
    opt.load_state_dict(copy.deepcopy(cp["optimizer_state"]))
    random.setstate(cp["rng_state"]["python"])
    torch.set_rng_state(cp["rng_state"]["torch"].cpu())
    if cp["rng_state"]["cuda"]:torch.cuda.set_rng_state_all([v.cpu() for v in cp["rng_state"]["cuda"]])
    gen=torch.Generator();gen.set_state(cp["loader_state"].cpu())
    exact(model.state_dict(),cp["model_state"],"model")
    exact(opt.state_dict(),cp["optimizer_state"],"optimizer")
    exact(native().rng_state(),cp["rng_state"],"rng")
    exact(gen.get_state(),cp["loader_state"],"loader")
    return gen
def install(model,condition,keep_mode=None):
    b=model.blocks[1];details={"condition":condition,"target_layer":2,"hidden_noise":False}
    if condition=="P1":
        parts=("drive.weight","drive.bias","visible_factor","mode_factor","mode_logits","out.weight")
        frozen=[]
        for name,p in b.named_parameters():
            if name in parts:p.requires_grad_(False);p.grad=None;frozen.append("blocks.1."+name)
        assert len(frozen)==6
        details["frozen_names"]=frozen
    elif condition in ("P2","P4"):
        b.local_coupling*=0.01;details["external_write_gain"]=0.01
    elif condition=="P3":
        assert model.M>=2 and keep_mode is not None
        original=b.forward
        mask=torch.zeros(model.M,device=next(model.parameters()).device);mask[keep_mode]=1
        def masked(hidden,mode_mask=None,renormalize_mode_mask=False,stats=None):
            assert mode_mask is None
            return original(hidden,mode_mask=mask,renormalize_mode_mask=False,stats=stats)
        b.forward=masked;b._audit_binary_mask=mask
        details["keep_mode_zero_based"]=keep_mode
    else:assert condition=="P0"
    model._audit_intervention=details
    return details
def ctr(x):return x-x.mean(-1,keepdim=True)
def arr(x):return x.detach().cpu().numpy()
def factors(model):
    out=[]
    for block in model.blocks:
        O=arr(block.out.weight).astype(np.float64);O-=O.mean(0,keepdims=True)
        V=arr(block.visible_factor).astype(np.float64)
        D=arr(block.drive.weight).astype(np.float64).reshape(block.M,8,64)
        Q=arr(block.mode_factor).astype(np.float64)
        g=arr(block.mode_logits.softmax(0)).astype(np.float64)*block.local_coupling
        mask=getattr(block,"_audit_binary_mask",None)
        if mask is not None:g*=arr(mask)
        out.append([(O,V[m],D[m].T@Q[m,s*8:(s+1)*8],float(g[m])) for m in range(block.M) for s in range(2)])
    return out
def branch_inner(a,b):
    return np.array([f[3]*g[3]*np.sum((f[0].T@g[0])*(f[1].T@g[1])*(f[2].T@g[2])) for f,g in zip(a,b)])
def coupling(fs,previous=None):
    d={}
    for layer in range(2):
        n=branch_inner(fs[layer],fs[layer])
        d["B%d_branch_norm"%layer]=np.sqrt(np.maximum(n,0))
        d["B%d_norm"%layer]=np.sqrt(max(n.sum(),0))
        if previous is not None:
            pn=branch_inner(previous[layer],previous[layer]);inn=branch_inner(previous[layer],fs[layer])
            den=np.sqrt(np.maximum(n*pn,0))
            d["B%d_branch_cos"%layer]=np.divide(inn,den,out=np.full_like(inn,np.nan),where=den>0)
            d["B%d_cos"%layer]=inn.sum()/np.sqrt(n.sum()*pn.sum()) if n.sum()*pn.sum()>0 else np.nan
    return d
def centroid(x,ix):
    c=x[ix].mean(1);return c-c.mean(0,keepdims=True)
def quantities(c):
    S=np.einsum("ymd,ynd->mn",c,c,optimize=True)/59
    fullS=S;S=S[1:,1:];d=np.diag(S).copy();D=float(d.sum())
    off=S.copy();np.fill_diagonal(off,0);I=float(off.sum())
    valid=(d/64>=1e-24)&(d/max(D,1e-300)>=WEAK)
    den=np.sqrt(np.maximum(np.outer(d,d),0));ok=np.outer(valid,valid)
    R=np.full_like(S,np.nan);np.divide(S,den,out=R,where=ok)
    return dict(S=fullS,d=d,D=D,I=I,rho=I/D if D/64>=1e-24 else np.nan,
                coherent=D+I,R=R,mode_valid=valid,B0=float(fullS[0,0]),Bcross=float(2*fullS[0,1:].sum()))
def joint_power(x):
    x=x.reshape(59,59,-1);j=x-x.mean(0,keepdims=True)-x.mean(1,keepdims=True)+x.mean((0,1),keepdims=True)
    return float(np.mean(np.sum(j*j,-1)))
@torch.no_grad()
def measure(model,cfg,tokens,roles,labels,tr,te,include_mixed=True,pilot=False,batch=256):
    model.eval();M=cfg["M"];b1,b2=model.blocks;d=64;q=15
    cap={};hs=[];fields={k:[] for k in ("raw","effective","gamma")};mixed=[];logits=[];scales=[]
    normsum={k:np.zeros(M) for k in ("raw","effective")}
    check={"LN1_reconstruction":0.,"LN2_reconstruction":0.,"query_kernel":0.,"mixed_polynomial":0.}
    kernel=next(model.parameters()).new_zeros(16,M)
    for dist in range(1,min(q,b2.sweep.H)+1):kernel[q-dist]=b2.sweep.lambdas**dist
    def projection(h,hom=False):
        n=h.shape[0]
        source=F.linear(h,b2.drive.weight,None if hom else b2.drive.bias).reshape(n,16,M,8)
        right=(source*kernel[None,:,:,None]).sum(1);aux=torch.cat((right,torch.zeros_like(right)),-1)
        return torch.einsum("bd,mdo->bmo",h[:,q],b2.visible_factor),torch.einsum("bma,mao->bmo",aux,b2.mode_factor)
    def writeprod(v):
        g=b2.local_coupling*b2.mode_logits.softmax(0)
        mask=getattr(b2,"_audit_binary_mask",None)
        if mask is not None:g=g*mask
        return F.linear(v*g[None,:,None],b2.out.weight,None)
    def pre(block,args,l):cap["layer"]=l;cap["h"+str(l)]=args[0]
    def post(block,args,out,l):cap["post"+str(l)]=out
    for l,b in enumerate(model.blocks):
        hs.extend([b.register_forward_pre_hook(lambda b,a,l=l:pre(b,a,l)),
                   b.register_forward_hook(lambda b,a,o,l=l:post(b,a,o,l))])
    def sweephook(sweep,args,out):
        l=cap["layer"];b=model.blocks[l];h=cap["h"+str(l)];right,left=out
        if l==0:
            v=torch.einsum("bpd,mdo->bpmo",h,b.visible_factor)
            z=torch.einsum("bpma,mao->bpmo",torch.cat((right,left),-1),b.mode_factor)
            g=b.local_coupling*b.mode_logits.softmax(0)
            cap["u0"]=F.linear(v*z*g[None,None,:,None],b.out.weight,None)
        else:
            v=torch.einsum("bd,mdo->bmo",h[:,q],b.visible_factor)
            z=torch.einsum("bma,mao->bmo",torch.cat((right[:,q],left[:,q]),-1),b.mode_factor)
            cap["u1"]=writeprod(v*z)
    hs.append(model.sweep.register_forward_hook(sweephook))
    def error(key,a,b):check[key]=max(check[key],float((a-b).abs().max())/max(1.,float(a.abs().max()),float(b.abs().max())))
    try:
        for start in range(0,len(labels),batch):
            z=model(tokens[start:start+batch],roles[start:start+batch]);logits.append(arr(z))
            h0,u0,H,U=cap["h0"],cap["u0"],cap["post0"],cap["u1"]
            r1=h0+u0.sum(2)+b1.out.bias
            s1=(ctr(r1).square().mean(-1,keepdim=True)+b1.norm.eps).sqrt()
            r2=H[:,q]+U.sum(1)+b2.out.bias
            s2=(ctr(r2).square().mean(-1,keepdim=True)+b2.norm.eps).sqrt()
            error("LN1_reconstruction",b1.norm(r1),H)
            error("LN2_reconstruction",b2.norm(r2),cap["post1"][:,q])
            raw_bg=H[:,q]+b2.out.bias
            phi=b2.norm.weight*ctr(U)/s2[:,None,:]
            eff_bg=b2.norm.weight*ctr(raw_bg)/s2+b2.norm.bias
            vals={"raw":torch.cat((raw_bg[:,None,:],U),1),
                  "effective":torch.cat((eff_bg[:,None,:],phi),1),
                  "gamma":b2.norm.weight*ctr(torch.cat((raw_bg[:,None,:],U),1))}
            for name,v in vals.items():fields[name].append(arr(v))
            normsum["raw"]+=arr(U.norm(dim=-1).sum(0));normsum["effective"]+=arr(phi.norm(dim=-1).sum(0))
            scales.append(arr(s2[:,0]))
            if include_mixed and M>=2:
                delta=b1.norm.weight*ctr(u0)/s1[:,:,None,:]
                hbase=b1.norm.weight*ctr(h0+b1.out.bias)/s1+b1.norm.bias
                dd=delta.permute(0,2,1,3).reshape(-1,16,d)
                vd,zd=projection(dd,hom=True)
                vd=vd.reshape(len(H),M,M,d);zd=zd.reshape(len(H),M,M,d)
                prod=vd.sum(1)*zd.sum(1)-(vd*zd).sum(1)
                mx=writeprod(prod);mixed.append(arr(mx))
                if pilot:
                    v0,z0=projection(hbase)
                    vv,zz=projection(H)
                    error("query_kernel",writeprod(vv*zz),U)
                    singles=sum((v0*zd[:,n]+vd[:,n]*z0+vd[:,n]*zd[:,n]) for n in range(M))
                    reconstructed=writeprod(v0*z0+singles+prod)
                    error("mixed_polynomial",reconstructed,U)
                    if M>=2:
                        pair=writeprod(vd[:,0]*zd[:,1]+vd[:,1]*zd[:,0])
                        four=writeprod((v0+vd[:,0]+vd[:,1])*(z0+zd[:,0]+zd[:,1]))-writeprod((v0+vd[:,0])*(z0+zd[:,0]))-writeprod((v0+vd[:,1])*(z0+zd[:,1]))+writeprod(v0*z0)
                        error("mixed_polynomial",pair,four)
    finally:
        for h in hs:h.remove()
    y=arr(labels);ix=np.argsort(y,kind="stable").reshape(59,59)
    arrays={};summary={}
    for view,chunks in fields.items():
        x=np.concatenate(chunks).astype(np.float64);c=centroid(x,ix);v=quantities(c)
        arrays[view+"_C"]=c
        for k,a in v.items():
            arrays[view+"_"+k]=np.asarray(a)
            if np.asarray(a).ndim==0:summary[view+"_"+k]=float(a)
        if pilot:
            ff=np.fft.fft2(x[:,1:].reshape(59,59,M,64),axes=(0,1),norm="ortho")
            k=np.arange(1,59);fp=np.sum(np.abs(ff[k,k])**2)/(59*59)
            check[view+"_parseval"]=abs(float(fp)-v["D"])/max(1.,abs(v["D"]))
    for name,value in normsum.items():arrays[name+"_mean_norm"]=value/len(y)
    scale=np.concatenate(scales);summary["LN2_scale_mean"]=float(scale.mean())
    summary["LN2_scale_rms"]=float(np.sqrt(np.mean(scale**2)))
    if mixed:
        x=np.concatenate(mixed).astype(np.float64);mc=centroid(x,ix);ysum=x.sum(1)
        tp=float(np.mean(np.sum(mc.sum(1)**2,axis=-1)));jp=joint_power(ysum)
        summary.update(mixed_task_power=tp,mixed_joint_power=jp,mixed_phi=tp/jp if jp/64>=1e-24 else np.nan)
        arrays["mixed_C"]=mc
    else:summary.update(mixed_task_power=0. if M==1 else np.nan,mixed_joint_power=0. if M==1 else np.nan,mixed_phi=np.nan)
    z=np.concatenate(logits);arrays["logits"]=z
    for name,idx in [("train",arr(tr)),("test",arr(te))]:
        zz=z[idx].astype(np.float64);yy=y[idx];pred=zz.argmax(1);correct=int((pred==yy).sum())
        mx=zz.max(1);ce=float(np.mean(mx+np.log(np.exp(zz-mx[:,None]).sum(1))-zz[np.arange(len(yy)),yy]))
        other=zz.copy();other[np.arange(len(yy)),yy]=-np.inf
        margin=zz[np.arange(len(yy)),yy]-other.max(1)
        arrays[name+"_margin"]=margin
        summary.update({name+"_correct":correct,name+"_n":len(idx),name+"_accuracy":correct/len(idx),name+"_ce":ce,
                        name+"_margin_mean":float(margin.mean())})
    model.train()
    return summary,arrays,check
