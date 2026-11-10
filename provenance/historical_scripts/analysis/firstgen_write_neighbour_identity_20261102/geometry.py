import numpy as np
import torch

P=59
N=P*P
A,B=np.divmod(np.arange(N),P)
step=np.arange(P)
INV=np.array([0]+[pow(s,-1,P) for s in range(1,P)])
da,db=np.indices((P,P))
DIRECTION=np.where(db==0,P,(-da*INV[db])%P).astype(int)
DIAG=np.arange(P)
K=5
TOL=1e-12

def joint(u):
    x=u-u.mean(-1,keepdims=True)
    full=x.reshape(P,P,-1)
    j=full-full.mean(0,keepdims=True)-full.mean(1,keepdims=True)+full.mean((0,1),keepdims=True)
    return np.ascontiguousarray(j.reshape(N,-1)),float(np.sum(x*x)/N)

def field_stats(j,raw_power):
    norms=np.einsum('ij,ij->i',j,j)
    power=float(norms.mean())
    assert power>1e-24*raw_power
    valid=norms>1e-24*raw_power
    z=np.zeros_like(j);z[valid]=j[valid]/np.sqrt(norms[valid,None])
    spec=np.abs(np.fft.fft2(j.reshape(P,P,-1),axes=(0,1),norm='ortho'))**2
    spectrum=spec.sum(-1); gram=j.T@j
    omega=float((spectrum*spectrum).sum()/np.sum(gram*gram))
    projected=np.fft.ifft2(spectrum).real
    projected=projected/projected[0,0]
    return z,valid,omega,power,projected

def cos_map(z,valid):
    f=np.fft.fft2(z.reshape(P,P,-1),axes=(0,1))
    totals=np.fft.ifft2(np.sum(np.abs(f)**2,axis=2)).real
    if valid.all(): return totals/N
    vf=np.fft.fft2(valid.reshape(P,P))
    numbers=np.rint(np.fft.ifft2(np.abs(vf)**2).real)
    return np.divide(totals,numbers,out=np.full((P,P),np.nan),where=numbers>0)

def neighbours(z,valid,pre=None):
    # Receives no labels, formula for the answer, scores, or train/test assignment.
    # If pre is supplied, rank identical ordered pairs by their increase in cosine.
    v=torch.from_numpy(np.ascontiguousarray(z))
    before=None if pre is None else torch.from_numpy(np.ascontiguousarray(pre))
    eligible=np.flatnonzero(valid)
    assert len(eligible)>K+1
    bad=torch.from_numpy(~valid)
    all_query=[]; all_ref=[]; all_w=[]; boundary_ties=0
    example=[]
    for start in range(0,len(eligible),192):
        ids=eligible[start:start+192]
        qi=torch.from_numpy(ids)
        matrix=v[qi]@v.T
        if before is not None: matrix-=before[qi]@before.T
        if not valid.all(): matrix[:,bad]=-np.inf
        matrix[torch.arange(len(ids)),qi]=-np.inf
        values,indices=torch.topk(matrix,K+1,dim=1,sorted=True)
        values=values.numpy();indices=indices.numpy()
        tie=(values[:,K-1]-values[:,K]<=TOL)
        boundary_ties+=int(tie.sum())
        all_query.append(np.repeat(ids[~tie],K));all_ref.append(indices[~tie,:K].reshape(-1))
        all_w.append(np.ones((~tie).sum()*K,dtype=np.float64))
        if np.any(tie):
            for i in np.flatnonzero(tie):
                row=matrix[i].numpy();cut=values[i,K-1]
                strong=np.flatnonzero(row>cut+TOL);eq=np.flatnonzero(abs(row-cut)<=TOL)
                weight=(K-len(strong))/len(eq)
                ref=np.r_[strong,eq];w=np.r_[np.ones(len(strong)),np.full(len(eq),weight)]
                assert 0.<weight<=1. and abs(w.sum()-K)<1e-10
                all_query.append(np.full(len(ref),ids[i]));all_ref.append(ref);all_w.append(w)
        sample=np.flatnonzero(ids==2*P+5)
        if len(sample):
            i=sample[0]
            example=[(int(ref),float(val)) for val,ref in zip(values[i,:K],indices[i,:K])]
    query=np.concatenate(all_query).astype(np.int32);ref=np.concatenate(all_ref).astype(np.int32)
    weights=np.concatenate(all_w)
    assert np.all(ref!=query) and abs(weights.sum()-len(eligible)*K)<1e-7
    return query,ref,weights,example,boundary_ties

def interpret(q,r,weight,test):
    # All input groups are assigned only after blind ranking.
    x=(A[r]-A[q])%P;y=(B[r]-B[q])%P
    code=DIRECTION[x,y]
    displacement=np.bincount(x*P+y,weights=weight,minlength=N).reshape(P,P)
    counts=np.bincount(code,weights=weight,minlength=P+1)
    total=counts.sum()
    assert np.isclose(displacement.sum(),total) and displacement[0,0]==0
    assert np.array_equal(code==1,((A[r]+B[r]-A[q]-B[q])%P)==0)
    assert np.array_equal(code==P-1,((A[r]-B[r]-A[q]+B[q])%P)==0)
    def share(mask):
        return float(weight[mask & (code==1)].sum()/weight[mask].sum()) if np.any(mask) else np.nan
    per_query=np.bincount(q,weights=weight*(code==1),minlength=N)/K
    return counts/total,displacement/total,per_query,share(test[q]),share(~test[q])

def grouped_cos(c):
    mask=(da+db!=0)
    return np.bincount(DIRECTION[mask],weights=c[mask],minlength=P+1)/(P-1)

def checks():
    rng=np.random.default_rng(20261102)
    j=rng.normal(size=(N,5));z,valid,_,_,g=field_stats(j,2.)
    c=cos_map(z,valid)
    for u,v in [(0,0),(1,-1),(1,1),(13,18)]:
        shifted=np.roll(z.reshape(P,P,-1),(-u,-v),(0,1)).reshape(N,-1)
        assert abs(c[u%P,v%P]-(z*shifted).sum(1).mean())<2e-12
    for slope in [1,58,3]:
        # Equal nearest scores deliberately exercise the fractional cutoff.
        Y=np.eye(P)[(A+slope*B)%P].astype(float)
        q,r,w,_,ties=neighbours(Y,np.ones(N,bool))
        fractions,_,_,_,_=interpret(q,r,w,(A%2==0))
        assert abs(fractions[slope]-1)<1e-10 and ties==N
        assert abs(grouped_cos(cos_map(Y,np.ones(N,bool)))[slope]-1)<1e-12
    return True

