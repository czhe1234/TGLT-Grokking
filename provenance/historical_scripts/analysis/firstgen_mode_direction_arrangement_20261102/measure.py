import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,hashlib,time,io
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
N=Path(__file__).resolve().parent;A=N.parent
O=A/'firstgen_coupling_state_structure_20261101';P=A/'firstgen_coupling_symmetry_prevalence_20261101'
T=A/'firstgen_physical_observables_table_20261101'
SRC=A.parent/'week/v1/code/exp2_sequence_local_grokking.py'
STEPS=np.arange(0,100001,1000)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def candidates(row):
 rid=row['run_id']
 if row['dataset']=='phase':return O/'existing_phase_check/runs'/f'{rid}_matrices.npz', O/'existing_phase_check/runs'/f'{rid}_manifest.csv'
 mat=O/'runs'/f'{rid}_matrices.npz'
 man=(O/'runs'/f'{rid}_manifest.csv') if mat.exists() else P/'runs'/f'original__{rid}__original_manifest.csv'
 return mat,man

def factors(sd,l,cfg):
 def ar(k):return sd[f'blocks.{l}.'+k].detach().double().cpu().numpy().copy()
 o=ar('out.weight');o-=o.mean(axis=0,keepdims=True)
 v=ar('visible_factor');m,d,_=v.shape;da=cfg['d_a']
 drive=ar('drive.weight').reshape(m,da,d);aux=ar('mode_factor')
 e=np.stack([[drive[j].T@aux[j,s*da:(s+1)*da] for s in range(2)] for j in range(m)])
 g=ar('mode_logits');g=np.exp(g-g.max());g=g/g.sum()*cfg['local_coupling']
 return o,v,e,g

def kernel(lam,H,L):
 distances=np.arange(1,min(H,L-1)+1,dtype=float)
 return (((L-distances)/L)[None,None,:]*(lam[:,None,None]*lam[None,:,None])**distances[None,None,:]).sum(2)
def coupling_gram(fs,kap):
 o,v,e,g=fs;m=len(g);oo=o.T@o;k=np.zeros((m,m))
 for i in range(m):
  for j in range(i,m):
   vv=v[i].T@v[j]
   value=sum(np.sum(oo*vv*(e[i,s].T@e[j,s])) for s in range(2))*g[i]*g[j]*kap[i,j]
   k[i,j]=k[j,i]=value
 return k

def load_cp(record):
 path=Path(record.get('checkpoint_path',record.get('path')));expected=record.get('checkpoint_sha256',record.get('sha256'))
 blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest();assert digest==expected,(path,digest,expected)
 cp=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);assert int(cp['step'])==int(record['step'])
 cfg=cp['config'];sd=cp['model_state'];assert cfg['layers']==2 and cfg['block_type']=='full' and not cfg['learnable_lambda']
 key=next(k for k in sd if k.endswith('raw_lambda'));lam=(torch.sigmoid(sd[key].double().cpu())*.999).numpy()
 assert sha(path)==expected
 return cp,lam,path,digest

def metrics(k):
 diag=np.diag(k);den=np.sqrt(diag)[:,None]*np.sqrt(diag)[None,:];valid=np.isfinite(diag)&(diag>0)
 r=np.divide(k,den,out=np.full_like(k,np.nan),where=den>0)
 assert np.all(valid),'invalid mode power'
 assert np.max(np.abs(r))<=1+2e-10
 assert np.max(np.abs(np.diag(r)-1))<1e-10
 assert np.linalg.eigvalsh(k).min()>-1e-10*max(1.,diag.sum())
 return r

def source_manifest(row):
 _,man=candidates(row);x=list(csv.DictReader(man.open()));dedup={int(a['step']):a for a in x};assert set(dedup)==set(STEPS)
 return dedup,man

def pilot(rows):
 lookup={x['run_id']:x for x in rows};results=[]
 points=[('r2_T4_M4_seed1',58000),('r2_T5_M2_seed2',0),('b0.25_wd0.1_s0',100000),('ref_b1_wd0.547722557505_s1',97000),('ref_b1_wd0.547722557505_s2',100000)]
 for rid,step in points:
  row=lookup[rid];records,man=source_manifest(row);cp,lam,path,digest=load_cp(records[step]);cfg=cp['config'];kap=kernel(lam,cfg['r']*cfg['T'],cfg['seq_len'])
  for l in range(2):
   fs=factors(cp['model_state'],l,cfg);k=coupling_gram(fs,kap);o,v,e,g=fs
   b=np.stack([[g[m]*np.einsum('ok,ak,bk->oab',o,v[m],e[m,s],optimize=True) for s in range(2)] for m in range(len(g))]);flat=b.reshape(len(g),-1)
   kd=(flat@flat.T)*kap;error=float(np.max(np.abs(k-kd))/max(1,float(np.max(np.abs(kd)))));assert error<1e-10
   assert np.max(np.abs(metrics(k)-metrics(kd)))<1e-9
   r=metrics(k);scale=np.exp(np.linspace(-30,30,len(g)));scaled=metrics(k*scale[:,None]*scale[None,:]);assert np.max(np.abs(r-scaled))<1e-10
   perm=np.random.default_rng(20261102).permutation(o.shape[0]);kp=coupling_gram((o[perm],v[:,perm,:],e[:,:,perm,:],g),kap);assert np.max(np.abs(k-kp))<1e-9*max(1,np.max(np.abs(k)))
   cached,_=candidates(row);cacheerr=None
   if cached.exists():
    z=np.load(cached);kk=z['matrices'][step//1000,l,0];cacheerr=float(np.max(np.abs(k-kk)));assert cacheerr<1e-9*max(1,np.max(np.abs(k)))
   results.append(dict(run_id=rid,step=step,layer=l+1,dense_relative_error=error,scale_invariant=True,coordinate_permutation_invariant=True,cached_error=cacheerr,checkpoint_sha256=digest))
 # Synthetic checks are not included in empirical results.
 for vectors,expected in [(np.eye(3),0.),(np.array([[1.,0.],[-1.,0.],[1.,0.]]),1.)]:
  rr=metrics(vectors@vectors.T);ii=np.triu_indices(len(vectors),1);assert abs(np.mean(rr[ii]**2)-expected)<1e-12
 save(N/'pilot_checks.json',dict(passed=True,results=results,synthetic_cases=2,cpu_only=True,training_updates=0))

def one(row):
 rid=row['run_id'];started=time.time();cache,man=candidates(row);records,_=source_manifest(row);hashes={str(man):sha(man),row['config_path']:sha(row['config_path'])}
 assert hashes[row['config_path']]==row['config_sha256']
 cp_reads=0;ms=[]
 if cache.exists():
  hashes[str(cache)]=sha(cache);z=np.load(cache);ks=z['matrices'][:,:,0].copy();assert ks.shape==(101,2,int(row['M']),int(row['M']))
  for st in STEPS:ms.append(dict(run_id=rid,step=int(st),checkpoint_path=records[st].get('checkpoint_path',records[st].get('path')),checkpoint_sha256=records[st].get('checkpoint_sha256',records[st].get('sha256')),source='validated existing K matrices; checkpoint hash from original manifest'))
 else:
  kk=[]
  for st in STEPS:
   cp,lam,path,digest=load_cp(records[st]);cfg=cp['config'];assert cfg['M']==int(row['M']);kap=kernel(lam,cfg['r']*cfg['T'],cfg['seq_len'])
   kk.append([coupling_gram(factors(cp['model_state'],l,cfg),kap) for l in range(2)])
   ms.append(dict(run_id=rid,step=int(st),checkpoint_path=str(path),checkpoint_sha256=digest,source='checkpoint read and before-after SHA256 verified'));cp_reads+=1
  ks=np.stack(kk)
 rs=np.stack([[metrics(k) for k in layers] for layers in ks])
 out=N/'arrays';out.mkdir(exist_ok=True);np.savez_compressed(out/f'{rid}.npz',steps=STEPS,K=ks,R=rs)
 for path,h in hashes.items():assert sha(path)==h
 return dict(run_id=rid,reused=cache.exists(),new_checkpoint_reads=cp_reads,seconds=time.time()-started,input_hashes=hashes,manifest=ms)

if __name__=='__main__':
 start=time.time();rows=list(csv.DictReader((N/'cohort.csv').open()));multi=[x for x in rows if int(x['M'])>1]
 save(N/'not_applicable.json',[dict(run_id=x['run_id'],reason='M=1: no distinct mode pair') for x in rows if int(x['M'])==1])
 if '--pilot' in sys.argv:pilot(multi);print('pilot passed',flush=True);sys.exit()
 assert json.loads((N/'pilot_checks.json').read_text())['passed']
 inputs=json.loads((N/'initial_input_hashes.json').read_text());inputs[str(SRC)]=sha(SRC);assert inputs[str(SRC)]=='e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73'
 for path,h in inputs.items():assert sha(path)==h
 done=[];man=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(one,row) for row in multi]
  for f in as_completed(jobs):
   d=f.result();inputs.update(d.pop('input_hashes'));man.extend(d.pop('manifest'));done.append(d);print(json.dumps(d),flush=True)
   save(N/'status.json',dict(completed=len(done),total=len(multi),seconds=time.time()-start))
 old=pd.read_csv(A/'firstgen_crossmode_coupling_overlap_20261101/curves.csv');maxerr=0.
 for row in multi:
  z=np.load(N/'arrays'/f'{row["run_id"]}.npz');k=z['K'][:,1];m=int(row['M']);mask=~np.eye(m,dtype=bool);diag=np.diagonal(k,axis1=1,axis2=2);den=np.sqrt(diag[:,:,None]*diag[:,None,:])[:,mask].sum(1);val=k[:,mask].sum(1)/den
  want=old[(old.dataset==row['dataset'])&(old.run_id==row['run_id'])].sort_values('step').coupling_overlap.to_numpy();assert len(want)==101
  e=np.max(np.abs(val-want));maxerr=max(maxerr,e);assert e<1e-9
 for path,h in inputs.items():assert sha(path)==h
 pd.DataFrame(man).to_csv(N/'point_manifest.csv',index=False)
 save(N/'input_hashes.json',inputs)
 save(N/'measurement_checks.json',dict(passed=True,cpu_only=True,training_updates=0,model_forwards=0,runs=len(done),checkpoints_represented=len(man),reused_runs=sum(x['reused'] for x in done),new_checkpoint_reads=sum(x['new_checkpoint_reads'] for x in done),old_signed_overlap_max_error=float(maxerr),inputs_unchanged=True,seconds=time.time()-start,details=done))
 print('all measurements complete',flush=True)
