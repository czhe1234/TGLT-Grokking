import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
from pathlib import Path
R=Path(__file__).resolve().parent;A=R.parent;S=A/'firstgen_write_neighbour_identity_20261102'
sys.path.insert(0,str(S))
import measure as old, geometry as geom
import numpy as np,pandas as pd,torch
import io,hashlib,json,argparse,time,inspect
from concurrent.futures import ProcessPoolExecutor,as_completed
P,N=59,3481;kk=np.arange(1,P);sl=np.arange(P);ii=sl[:,None]*kk%P
torch.set_num_threads(1)
def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,d):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)
def read(p):return json.loads(p.read_text())

def line_energies(field):
    f=np.fft.fft2(field.reshape(P,P,-1),axes=(0,1),norm='ortho')
    g=(np.abs(f)**2).sum(-1)
    family=g[kk[None,:],ii].sum(1)
    return np.r_[family,g[0,kk].sum()]+g[0,0]

def continuous(write):
    j,power0=geom.joint(write)
    lengths=np.einsum('ij,ij->i',j,j)
    good=lengths>1e-24*power0
    if lengths.mean()<=1e-24*power0 or good.sum()<2:return np.full(60,np.nan),good,None
    z=np.zeros_like(j);z[good]=j[good]/np.sqrt(lengths[good,None])
    valid_count=good.sum()
    numerator=P*line_energies(z)-valid_count
    if good.all(): denominator=np.full(P+1,N*(P-1))
    else:denominator=np.rint(P*line_energies(good.astype(float)[:,None])-valid_count).astype(np.int64)
    cs=np.divide(numerator,denominator,out=np.full(P+1,np.nan),where=denominator>0)
    assert np.nanmin(cs)>=-1-2e-10 and np.nanmax(cs)<=1+2e-10
    return cs,good,z

def direct(z,valid):
    # Independent real-space sum over groups. No FFT, nearest-neighbour or rank operations.
    ans=[];inds=np.arange(P)
    for slope in range(P+1):
        if slope==P: ids=(inds[None,:]*P+inds[:,None])
        else: ids=((inds[:,None]-slope*inds[None,:])%P)*P+inds[None,:]
        cnt=valid[ids].sum(1);den=sum(cnt*(cnt-1))
        vec=z[ids].sum(1);ans.append(((vec*vec).sum()-valid.sum())/den if den else np.nan)
    return np.array(ans)
def model_setup(rid):
    meta=paths[rid];folder=Path(meta['folder']);cfg=read(folder/'config.json')
    assert sha((folder/'config.json').read_bytes())==meta['config_sha'] and sha((folder/'split.json').read_bytes())==meta['split_sha']
    native=old.native
    kw={k:v for k,v in cfg.items() if k in inspect.signature(native.SequenceLocalModeModel).parameters}
    kw['lambdas']=cfg.get('lambda_values') or native.xi_spread_lambdas(cfg['M'],cfg['xi_min'],cfg['xi_max']).tolist()
    model=native.SequenceLocalModeModel(**kw).double().eval()
    tok,roles,labels,_=native.build_sequence_data(59,'add',16,0,7,15)
    test=np.zeros(N,bool);test[read(folder/'split.json')['test_idx']]=True
    return model,cfg,tok,roles,labels.numpy(),test
def replay(rid,pt,state,full=False):
    model,cfg,tok,roles,lab,test=state
    raw=Path(pt.checkpoint).read_bytes();assert sha(raw)==pt.sha256
    cp=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    assert cp['step']==pt.step and all(cp['config'][k]==cfg[k] for k in ('seed','wd','local_coupling','r','T','M','d','d_a','task','train_fraction','split_seed','seq_len','pos_a','pos_b','query_pos','layers'))
    model.load_state_dict(cp['model_state'],strict=True)
    init=''
    if pt.step==0:
        dh=hashlib.sha256()
        for k,v in sorted(cp['model_state'].items()):dh.update(k.encode());dh.update(v.contiguous().numpy().tobytes())
        init=dh.hexdigest()
    u,log=old.write_exact(model,tok,roles)
    cs,mask,z=continuous(u)
    correct=log.argmax(1)==lab
    diff=int(correct[test].sum())-round(pt.test_acc*test.sum());tdiff=int(correct[~test].sum())-round(pt.train_acc*(~test).sum())
    check=dict(diff=diff,train_diff=tdiff,invalid=int(N-mask.sum()),init=init)
    if full or max(abs(diff),abs(tdiff))>2:
        fields,native_logits,_=old.prev.capture(model,cfg,tok,roles)
        uu=fields['raw_write'].sum(1)
        uscale=np.linalg.norm(uu)
        relerr=np.linalg.norm(u-u.mean(-1,keepdims=True)-uu)/(uscale if uscale else 1.)
        le=np.max(np.abs(log-native_logits));assert relerr<1e-8 and le<1e-8,(rid,pt.step,relerr,le)
        check.update(native_write_error=float(relerr),native_logit_error=float(le))
    if full and z is not None:
        check['direct_error']=float(np.max(abs(direct(z,mask)-cs)))
        check['old_convention_error']=float(np.max(abs(geom.grouped_cos(geom.cos_map(z,mask))-cs)))
        assert max(check['direct_error'],check['old_convention_error'])<1e-10
        # Each held-out input affects neither which shift family is tested nor how it is weighted.
        cs2=continuous(13.*u)[0];assert np.max(abs(cs2-cs))<1e-10
    assert sha(Path(pt.checkpoint).read_bytes())==pt.sha256
    return cs,check

def summary(cs):
    if not np.isfinite(cs).all():return np.full(4,np.nan)
    other=(cs.sum()-cs[1])/P
    return cs[1],cs.std(ddof=0),other,cs[1]-other
def chunk(job):
    rid,lo,hi=job;points=group[rid];state=None
    cs_arr=[];stats=[];times=[]
    for pt in points[lo:hi]:
        is_cache=pt.source=='prior_continuous_correlations'
        if not is_cache or pt.step==0:
            if state is None:state=model_setup(rid)
            cs,flags=replay(rid,pt,state)
            if is_cache:
                assert np.max(abs(cs-cached[rid][pt.step//cache_dt[rid]]))<1e-10
        else:
            assert sha(Path(pt.checkpoint).read_bytes())==pt.sha256
            cs=cached[rid][pt.step//cache_dt[rid]]
            stat=cached_stats.loc[(rid,pt.step)];assert abs(cs[1]-stat.cos_same_sum)<1e-10 and stat.invalid_inputs==0
            flags=dict(diff=int(stat.correct_difference),train_diff=None,invalid=0,init='')
        c,sd,o,ct=summary(cs)
        stats.append(dict(run_id=rid,step=pt.step,C_add=c,C_sd=sd,C_other=o,C_contrast=ct,train_acc=pt.train_acc,test_acc=pt.test_acc,source=pt.source,**flags));cs_arr.append(cs);times.append(pt.step)
    base=R/'parts'/f'{rid}__{lo:04d}'
    pd.DataFrame(stats).to_csv(str(base)+'.csv',index=False);np.savez_compressed(str(base)+'.npz',step=times,correlations=cs_arr)
    return len(times)
paths=read(R/'provenance.json')
manifest=pd.read_csv(R/'point_manifest.csv.gz')
group={k:list(v.itertuples(index=False)) for k,v in manifest.groupby('run_id',sort=False)}
info=read(S/'roster.json');cache_dt={k:v['dt'] for k,v in info.items()}
cached_stats=pd.read_csv(S/'neighbour_trajectories.csv.gz').set_index(['run_id','step'])
with np.load(S/'all_60_directions.npz') as z:
    cached={k:z[k][:,1,:].copy() for k in paths if k in z.files}
def piloteval():
    display=[]
    targets=[('b0.25_wd0.01_s0',0),('b1_wd0.1_s0',75000),('b4_wd1_s2',100000),('ref_b0.25_wd0.547722557505_s1',200000),('prior_r2_p59_add_c000_seed0',50000),('prior_r2_p59_add_c035_seed0',25000)]
    for rid,st in targets:
        point=next(q for q in group[rid] if q.step==st);cs,stats=replay(rid,point,model_setup(rid),full=True)
        if point.source.startswith('prior'):stats['cache_error']=float(np.max(abs(cs-cached[rid][st//cache_dt[rid]])));assert stats['cache_error']<1e-10
        display.append(dict(run_id=rid,step=st,**stats))
    rng=np.random.default_rng(347)
    # A partial-validity test, independent of actual activations.
    z=rng.normal(size=(N,8));valid=rng.random(N)>.1;z/=np.linalg.norm(z,axis=1,keepdims=True);z[~valid]=0.
    n=valid.sum();a=(P*line_energies(z)-n)/np.rint(P*line_energies(valid.astype(float)[:,None])-n)
    assert np.max(abs(a-direct(z,valid)))<1e-10
    yy=(geom.A+geom.B)%P
    rr=continuous(np.eye(P)[yy])[0];assert abs(rr[1]-1)<1e-10 and abs(np.mean(np.delete(rr,1))+1/(P-1))<1e-10
    return display
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--workers',type=int,default=20);args=parser.parse_args()
    fingerprints={x:sha((R/x).read_bytes()) for x in ['protocol.json','prepare.py','measure_beta.py','point_manifest.csv.gz','cohort.csv','provenance.json']}
    if args.pilot:dump(R/'pilot.json',dict(hashes=fingerprints,checks=piloteval()));print('pilot passed')
    else:
        assert read(R/'pilot.json')['hashes']==fingerprints
        jobs=[(rid,i,min(i+35,len(v))) for rid,v in group.items() for i in range(0,len(v),35)]
        jobs.sort(key=lambda j: (j[1],j[0]));done=0;start=time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            futures={ex.submit(chunk,j):j for j in jobs}
            for fut in as_completed(futures):
                try:done+=fut.result()
                except Exception as e:
                    dump(R/'error.json',dict(job=futures[fut],error=repr(e)));raise
                dump(R/'status.json',dict(done=done,total=len(manifest),elapsed_seconds=round(time.time()-start,1),complete=done==len(manifest)))
        print('complete',done)
