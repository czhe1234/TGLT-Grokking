"""Use exactly the same label-balanced logit-space force on the original training rows in both seeds."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"relay_learning_20261118"))
import replay as old
from order import cycles,P

times=[0,4000,10000,20000,40000,60000,85000,91500,105500,107000,155000,200000,245000,256000,296000]
rng=torch.Generator().manual_seed(119)
wrong=old.Y.clone()
wrong[old.TR]=old.Y[old.TR][torch.randperm(len(old.TR),generator=rng)]
real,sham=cycles("cpu",20261119)


def class_weights(labels,ix):
    count=torch.bincount(labels[ix],minlength=P)
    assert (count>0).all()
    return 1/(count[labels[ix]].double()*P)


def score(z,y,weights):
    return ((z.gather(1,y[:,None]).flatten()-z.mean(1))*weights).sum()


def gradients(m,ix,labels,names):
    ps=list(names.values())
    total=[torch.zeros_like(p) for p in ps]
    weights=class_weights(labels,ix)
    for off in range(0,len(ix),192):
        batch=ix[off:off+192]
        h=old.bridge.first_layer(m,old.DATA["tokens"][batch],old.DATA["roles"][batch],"TGLT")
        z=old.bridge.native_parts(m,h,"TGLT")[0]
        s=score(z,labels[batch],weights[off:off+len(batch)])
        for v,g in zip(total,torch.autograd.grad(s,ps)):
            v.add_(g)
    return dict(zip(names,total))


@torch.no_grad()
def observed(m):
    s=old.capture(m)
    bl=m.blocks[1]
    v=torch.einsum("bd,mdo->bmo",s["h"][:P,15],bl.visible_factor)
    v-=v.mean(0)
    x=s["received2"].reshape(P,P,4,8)
    f=torch.einsum("nmc,mco->nmo",(x-x.mean(0)).reshape(P*P,4,8),bl.mode_factor[:,:8])
    q=F.linear((bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]*v[old.B]*f).sum(1),bl.out.weight)
    removed=old.at_head(m,s["pre2"]-q)
    return {name:float(score(z[ix],old.Y[ix],class_weights(old.Y,ix))) for name,z,ix in
            (("train",s["z"],old.TR),("test",s["z"],old.TE),("removed",removed,old.TE))}


def run(job):
    rid,t=job
    m,_,_,_,_=old.load(rid,t)
    names={n:p for n,p in m.named_parameters() if p.requires_grad}
    a=gradients(m,old.TR,old.Y,names)
    null=gradients(m,old.TR,wrong,names)
    b=gradients(m,old.TE,old.Y,names)
    scopes=dict(all=list(names),numerals=["residue.weight"],first=[n for n in names if n.startswith("blocks.0.")],
                second=[n for n in names if n.startswith("blocks.1.")],
                classifier=[n for n in names if n.startswith("classifier.")])
    rows=[]
    for name,keys in scopes.items():
        def vec(d):
            # The numeral-only stimulus excludes padding. In the full model padding is a normal parameter.
            return torch.cat([(d[k][1:] if name=="numerals" else d[k]).flatten() for k in keys])
        te=vec(b)
        for force,gr in (("true",a),("shuffled",null)):
            tr=vec(gr)
            length=tr.norm()
            cross=tr@te
            row=dict(run_id=rid,step=t,scope=name,force=force,train_norm=float(length),test_norm=float(te.norm()),
                     transfer=float(cross/length.square()),cosine=float(cross/(te.norm()*length)))
            if name=="numerals":
                _,g=real.measure(m.residue.weight[1:].detach(),True)
                u=g.flatten()/g.norm()
                row.update(train_order_cosine=float(u@tr/length),test_order_cosine=float(u@te/te.norm()))
            if name in ("all","numerals") and force=="true" and t in (4000,40000,85000,105500,155000,256000):
                with torch.no_grad():
                    base={k:names[k].clone() for k in keys}
                    eps=1e-6
                    conditions=[]
                    for sign in (-1,1):
                        for k in keys:
                            names[k].copy_(base[k])
                            if name=="numerals":
                                names[k][1:].add_(gr[k][1:],alpha=sign*eps/float(length))
                            else:
                                names[k].add_(gr[k],alpha=sign*eps/float(length))
                        conditions.append(observed(m))
                    for k in keys:
                        names[k].copy_(base[k])
                    dz={k:(conditions[1][k]-conditions[0][k])/(2*eps) for k in conditions[0]}
                    assert abs(dz["train"]-float(length))<2e-4*(1+float(length)),(rid,t,name,dz)
                    assert abs(dz["test"]/dz["train"]-row["transfer"])<3e-5,(rid,t,name,row,dz)
                    row.update(finite_transfer=dz["test"]/dz["train"],finite_no_query_transfer=dz["removed"]/dz["train"])
            rows.append(row)
    return rows

if __name__=="__main__":
    jobs=[(r,t) for r in old.RUNS[:2] for t in times]
    with ProcessPoolExecutor(max_workers=5) as pool:
        rows=[row for group in pool.map(run,jobs) for row in group]
    old.table(HERE/"results/training_response.csv",rows)
    print(len(rows),flush=True)
