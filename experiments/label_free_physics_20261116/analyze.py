"""Label-free fields and exact input response. Existing checkpoints; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(name,'2')
import argparse,copy,csv,hashlib,importlib.util,json,math,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=Path(os.environ.get('TGLT_SOURCE_ROOT','/home/chenzhe/work/tglt_experiments'))
spec=importlib.util.spec_from_file_location('firstgen_observable_reference',REPO/'src/exp2_sequence_local_grokking.py')
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)


def dump(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)


def write_csv(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def input_grid(pos_b,query):
    # Construct inputs only. No modular addition, labels, split, or classifier.
    tokens=torch.zeros(59*59,16,dtype=torch.long)
    tokens[:,0]=torch.arange(59).repeat_interleave(59)+1
    tokens[:,pos_b]=torch.arange(59).repeat(59)+1
    roles=torch.zeros_like(tokens);roles[:,0]=1;roles[:,pos_b]=2;roles[:,query]=3
    return tokens,roles


class DenseSweep(nn.Module):
    def __init__(self,right,left):
        super().__init__();self.register_buffer('K_right',right);self.register_buffer('K_left',left)
    def forward(self,source):
        return tuple(torch.einsum('mij,bjmc->bimc',k,source) for k in [self.K_right,self.K_left])


def load_model(row,step):
    path=Path(row['path'])/('checkpoint_%07d.pt'%step)
    cp=torch.load(path,map_location='cpu',weights_only=False)
    state=cp['model_state'];r,T=(2,4) if row['span']==15 else (1,8)
    model=ref.SequenceLocalModeModel(p=59,seq_len=16,query_pos=row['query'],d=64,d_a=8,M=4,r=r,T=T,
        layers=2,lambdas=ref.xi_spread_lambdas(4,.7,32.),learnable_lambda=False,local_coupling=1.)
    extra={k:v for k,v in state.items() if k.endswith(('.K_right','.K_left','.mass_right','.mass_left'))}
    model.load_state_dict({k:v for k,v in state.items() if k not in extra})
    model=model.double().eval()
    # Old runs recompute the kernel in float64; new runs preserve their saved float32 kernel.
    if extra:
        right=state['sweep.K_right'].double();left=state['sweep.K_left'].double()
        mass_error=max(float((right.sum(-1)-state['sweep.mass_right'].double()).abs().max()),
                       float((left.sum(-1)-state['sweep.mass_left'].double()).abs().max()))
        assert mass_error<2e-6
    else:
        ix=torch.arange(16);distance=ix[:,None]-ix[None,:];lam=model.sweep.lambdas.detach()
        power=lam[:,None,None]**distance.abs()[None].clamp_min(1)
        right=power*((distance>0)&(distance<=8))[None]
        left=power*((distance<0)&(-distance<=8))[None]
    assert not bool(right[:,:,0][:,9:].any())
    native_sweep=model.sweep
    dense=DenseSweep(right,left)
    model.sweep=dense
    for block in model.blocks:block.sweep=dense
    model.requires_grad_(False)
    return model,native_sweep,path


def block_parts(block,hidden):
    n,length,d=hidden.shape
    source=block.drive(hidden).reshape(n,length,4,8)
    mode_state=torch.cat(block.sweep(source),dim=-1)
    visible=torch.einsum('bpd,mdo->bpmo',hidden,block.visible_factor)
    received=torch.einsum('bpma,mao->bpmo',mode_state,block.mode_factor)
    contribution=block.local_coupling*visible*received*block.mode_logits.softmax(0)[None,None,:,None]
    pre=hidden+block.out(contribution.sum(2))
    return block.norm(pre),pre,visible,contribution


def fields(model,tokens,roles):
    h0=model.residue_features(tokens)+model.role_features(roles)+model.position[None]
    h1,p1,v1,_=block_parts(model.blocks[0],h0)
    h2,p2,v2,c2=block_parts(model.blocks[1],h1)
    writes=F.linear(c2[:,model.query_pos],model.blocks[1].out.weight,None)
    return h0,h1,h2,p1,p2,v1,v2,writes


def layernorm_tangent(block,pre,J):
    centered=pre-pre.mean(-1,keepdim=True)
    variance=centered.square().mean(-1,keepdim=True)+block.norm.eps
    J=J-J.mean(-2,keepdim=True)
    J=J-centered[:,:,None]*torch.einsum('bi,bik->bk',centered,J)[:,None,:]/(pre.shape[-1]*variance[:,:,None])
    return J*block.norm.weight[None,:,None]/variance.sqrt()[:,:,None]


def exact_response(model,values):
    """Exact 64x64 Jacobian via two-layer chain rule, for q>H and 0<q-H<=H.

    h1[q] is independent of h0[0]. Every path passes through j in [q-H,H].
    Keep each layer's LayerNorm derivative; no frozen-denominator approximation.
    """
    h0,h1,h2,p1,p2,v1,v2,w=values
    q=model.query_pos;b1,b2=model.blocks;H=8
    assert H<q<=2*H
    first=b1.mode_factor[:,:8,:].transpose(-1,-2)@b1.drive.weight.reshape(4,8,64)
    received_tangent=torch.zeros(len(h0),4,8,64,dtype=h0.dtype)
    for j in range(q-H,H+1):
        factor=b1.mode_logits.softmax(0)*b1.local_coupling*model.sweep.K_right[:,j,0]
        pre_J=torch.einsum('bmo,mok,m->bok',v1[:,j],first,factor)
        pre_J=b1.out.weight[None]@pre_J
        J1=layernorm_tangent(b1,p1[:,j],pre_J)
        moved=torch.einsum('mci,bik->bmck',b2.drive.weight.reshape(4,8,64),J1)
        received_tangent+=moved*model.sweep.K_right[:,q,j][None,:,None,None]
    mode_J=torch.einsum('bmck,mco->bmok',received_tangent,b2.mode_factor[:,:8,:])
    pre_J=(mode_J*v2[:,q,:,:,None]*b2.mode_logits.softmax(0)[None,:,None,None]*b2.local_coupling).sum(1)
    pre_J=b2.out.weight[None]@pre_J
    J=layernorm_tangent(b2,p2[:,q],pre_J)
    return J


def centered_corr(a,b):
    ac=a-a.mean(0);bc=b-b.mean(0)
    va=float(ac.square().sum()/len(ac));vb=float(bc.square().sum()/len(bc))
    if va<1e-24 or vb<1e-24:return None,va,vb
    return float((ac*bc).sum()/len(ac)/math.sqrt(va*vb)),va,vb


def spectrum(a):
    ac=a-a.mean(0);eigenvalues=torch.linalg.eigvalsh(ac.T@ac/len(ac))
    assert float(eigenvalues.min())>=-1e-9*max(1.,float(eigenvalues.max()))
    eigenvalues=eigenvalues.clamp_min(0);trace=float(eigenvalues.sum())
    if trace<1e-24:return None,None,trace
    p=eigenvalues/eigenvalues.sum();positive=p[p>0]
    return float(-(positive*positive.log()).sum()),float(1/p.square().sum()),trace


def measure(row,step,batch_size=256):
    started=time.time();model,native,path=load_model(row,step)
    tokens,roles=input_grid(row['pos_b'],row['query'])
    collected={k:[] for k in ['h1_0','h1_q','h1_b','h2_0','h2_q','h2_b','total','mode3','mode4','chi']}
    with torch.no_grad():
        for start in range(0,len(tokens),batch_size):
            values=fields(model,tokens[start:start+batch_size],roles[start:start+batch_size])
            h0,h1,h2,p1,p2,v1,v2,w=values
            for level,h in [(1,h1),(2,h2)]:
                for suffix,position in [('0',0),('q',row['query']),('b',row['pos_b'])]:
                    collected['h%d_%s'%(level,suffix)].append(h[:,position].clone())
            for key,value in [('total',w.sum(1)),('mode3',w[:,2]),('mode4',w[:,3])]:collected[key].append(value.clone())
            J=exact_response(model,values);collected['chi'].append(J.square().sum((-2,-1)).sqrt())
    data={k:torch.cat(v) for k,v in collected.items()}
    result=dict(run_id=row['run_id'],span=row['span'],seed=row['seed'],step=step,input_count=len(tokens))
    for level in [1,2]:
        for suffix in ['q','b']:
            value,va,vb=centered_corr(data['h%d_0'%level],data['h%d_%s'%(level,suffix)])
            result['G%d_0%s'%(level,suffix)]=value
            result['variance_h%d_0'%level]=va;result['variance_h%d_%s'%(level,suffix)]=vb
    for name in ['total','mode3','mode4']:
        S,D,trace=spectrum(data[name]);result['S_'+name]=S;result['D_'+name]=D;result['variance_'+name]=trace
    result['q34'],_,_=centered_corr(data['mode3'],data['mode4'])
    chi=data['chi'];result.update(chi_mean=float(chi.mean()),chi_rms=float(chi.square().mean().sqrt()),
        chi_p10=float(torch.quantile(chi,.1)),chi_p90=float(torch.quantile(chi,.9)))
    result['seconds']=time.time()-started
    source=dict(run_id=row['run_id'],step=step,path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    return result,source


def roster():
    rows=json.loads((SOURCE/'experiments/firstgen_followups_20261114/mode_mechanism/roster.json').read_text())
    selected=[r for r in rows if r['H']==8 and (r['span']==11 or r['seed'] in [0,1])]
    for r in selected:
        r['pos_b']=7 if r['span']==15 else 5;r['query']=r['span']
        r['steps']=sorted(int(p.stem.split('_')[-1]) for p in Path(r['path']).glob('checkpoint_*.pt') if int(p.stem.split('_')[-1])<=40000)
    assert len(selected)==5
    return selected


def validate():
    results=[]
    for row in roster():
        for step in [0,20000,40000]:
            model,native,path=load_model(row,step);tokens,roles=input_grid(row['pos_b'],row['query'])
            ids=torch.tensor([0,1,59,1771,3480]);t,r=tokens[ids],roles[ids]
            with torch.no_grad():
                vals=fields(model,t,r);h0,h1,h2=vals[:3]
                assert float((h1-model.blocks[0](h0)).abs().max())<1e-10
                assert float((h2-model.blocks[1](h1)).abs().max())<1e-10
                if row['span']==15:
                    block=copy.deepcopy(model.blocks[0]);block.sweep=native
                    iterative_error=float((h1-block(h0)).abs().max());assert iterative_error<1e-10
                else:
                    cp=torch.load(path,map_location='cpu',weights_only=False)
                    saved=cp['model_state']['sweep.K_right'].double()
                    iterative_error=float((saved-model.sweep.K_right).abs().max());assert iterative_error==0
                exact=exact_response(model,vals)
            x=h0[0].detach().clone().requires_grad_(True)
            def forward_one(site0):
                hidden=torch.cat([site0[None],x[1:]],0)[None]
                for block in model.blocks:hidden=block(hidden)
                return hidden[0,row['query']]
            auto=torch.autograd.functional.jacobian(forward_one,x[0],vectorize=True)
            jac_error=float((auto-exact[0]).abs().max());assert jac_error<1e-9
            direction=torch.arange(64,dtype=torch.float64).sin();direction/=direction.norm();eps=1e-5
            finite=(forward_one(x[0]+eps*direction)-forward_one(x[0]-eps*direction))/(2*eps)
            fd_error=float((finite-exact[0]@direction).abs().max());assert fd_error<1e-6
            # A one-layer perturbation cannot cross q>8: this must be exactly zero.
            one=lambda site0:model.blocks[0](torch.cat([site0[None],x[1:]],0)[None])[0,row['query']]
            one_j=torch.autograd.functional.jacobian(one,x[0],vectorize=True)
            assert float(one_j.abs().max())==0
            results.append(dict(run_id=row['run_id'],step=step,forward_error=iterative_error,
                jacobian_max_error=jac_error,finite_difference_max_error=fd_error,first_layer_query_jacobian=0.))
    dump(dict(cpu_only=True,label_free=True,input_count=3481,checks=results),HERE/'validation/implementation_checks.json')
    print(json.dumps({'validation_cases':len(results),'jacobian_max_error':max(r['jacobian_max_error'] for r in results)},ensure_ascii=False),flush=True)


def run_one(row):
    out=HERE/'records'/row['run_id'];out.mkdir(parents=True,exist_ok=True)
    curves=[];sources=[]
    for step in row['steps']:
        result,source=measure(row,step);curves.append(result);sources.append(source)
        if step%5000==0:
            write_csv(curves,out/'curves.csv');write_csv(sources,out/'sources.csv')
            print(json.dumps(dict(run=row['run_id'],step=step,seconds=round(result['seconds'],3)),ensure_ascii=False),flush=True)
    write_csv(curves,out/'curves.csv');write_csv(sources,out/'sources.csv')
    return row['run_id'],len(curves)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate',action='store_true');ap.add_argument('--benchmark',action='store_true');ap.add_argument('--workers',type=int,default=5);args=ap.parse_args()
    rows=roster();dump(rows,HERE/'roster.json')
    if args.validate:validate();return
    if args.benchmark:
        result,source=measure(rows[0],20000);dump(result,HERE/'validation/benchmark.json');print(json.dumps(result,ensure_ascii=False),flush=True);return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_one,row) for row in rows]):print('FINISHED',f.result(),flush=True)
    allrows=[];sources=[]
    for row in rows:
        for name,dest in [('curves.csv',allrows),('sources.csv',sources)]:
            with (HERE/'records'/row['run_id']/name).open() as f:dest.extend(csv.DictReader(f))
    write_csv(allrows,HERE/'curves.csv');write_csv(sources,HERE/'sources.csv')
    dump(dict(checkpoints=len(allrows),runs=len(rows),missing=0,cpu_only=True,label_free=True),HERE/'validation/completion_checks.json')
    print('COMPLETE',len(allrows),flush=True)


if __name__=='__main__':main()
