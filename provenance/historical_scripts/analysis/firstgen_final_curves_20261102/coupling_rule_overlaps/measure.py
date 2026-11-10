"""Unnormalized and normalized parameter-only overlaps for all six mode pairs."""
import os
for key in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,gzip,hashlib,io,itertools,json,sys,time
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import numpy as np
import torch
torch.set_num_threads(1);sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent
analysis=out.parents[1]
old=analysis/'firstgen_mode_direction_arrangement_20261102'
campaign=analysis/'firstgen_beta_wd_gray_continuation_20261102'
grid=campaign/'results/seed_grids_100k'
spec=importlib.util.spec_from_file_location('frozen_coupling_formula',old/'measure.py')
formula=importlib.util.module_from_spec(spec);spec.loader.exec_module(formula)
proof=json.loads((grid/'verification.json').read_text())
manifest={r['run_id']:r for r in proof['source']}
assert len(manifest)==105 and formula.sha(formula.SRC)=='e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73'
pairs=list(itertools.combinations(range(4),2))
steps=list(range(0,100001,500))
cols=['run','wd','beta','seed','step','mean_R','mean_K','squared_norm_of_mean_unit_rule']+[f'R{i+1}{j+1}' for i,j in pairs]+[f'K{i+1}{j+1}' for i,j in pairs]+[f'norm{i+1}' for i in range(4)]
def run(rid):
 r=manifest[rid]; folder=Path(r['original_history']).parent
 dest=out/'parts'/f'{rid}.csv'
 if dest.exists():return rid
 previous=old/'arrays'/f'{rid}.npz'; legacy={}
 if previous.exists():
  with np.load(previous) as archive:
   assert archive['R'].shape==(101,2,4,4)
   legacy={int(t):(k[1],m[1]) for t,k,m in zip(archive['steps'],archive['K'],archive['R'])}
 rows=[]; err=0.
 for st in steps:
  p=folder/f'checkpoint_{st:07d}.pt'
  receipt=json.loads((folder/f'receipt_{st:07d}.json').read_text())
  raw=p.read_bytes();assert receipt['step']==st and len(raw)==receipt['checkpoint_bytes'] and hashlib.sha256(raw).hexdigest()==receipt['checkpoint_sha256']
  cp=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);cfg=cp['config'];sd=cp['model_state']
  assert cp['step']==st and (cfg['seed'],cfg['local_coupling'],cfg['wd'])==(r['seed'],r['beta'],r['wd'])
  assert (cfg['p'],cfg['M'],cfg['layers'],cfg['r'],cfg['T'],cfg['seq_len'],cfg['block_type'],cfg['learnable_lambda'])==(59,4,2,3,4,16,'full',False)
  keys=[k for k in sd if k.endswith('raw_lambda')]
  lam=(torch.sigmoid(sd[keys[0]].double())*.999).numpy();assert all(torch.equal(sd[keys[0]],sd[key]) for key in keys)
  kap=formula.kernel(lam,cfg['r']*cfg['T'],cfg['seq_len'])
  fs=formula.factors(sd,1,cfg); K=formula.coupling_gram(fs,kap); R=formula.metrics(K)
  upper=np.triu_indices(4,k=1); av=float(R[upper].mean()); rawav=float(K[upper].mean())
  q=float(R.sum()/16);assert abs(q-(1+3*av)/4)<2e-12 and -1e-12<=q<=1+1e-12
  if st in legacy:
   oldK,oldR=legacy[st]
   assert np.allclose(K,oldK,rtol=1e-10,atol=1e-11) and np.max(np.abs(R-oldR))<1e-9
  if rid=='b0.25_wd0.1_s0' and st in (0,50000,100000):
   # Independent dense construction including distance and both propagation directions.
   om,vis,e,gates=fs
   flat=np.stack([[gates[m]*np.einsum('ok,ik,jk->oij',om,vis[m],e[m,direction],optimize=True) for direction in range(2)] for m in range(4)]).reshape(4,-1)
   expect=(flat@flat.T)*kap; ref=formula.metrics(expect)
   assert np.max(np.abs(R-ref))<1e-9 and np.allclose(K,expect,rtol=1e-9,atol=1e-12)
   # Direct four-way addition for the normalized coefficient tables, without pair averaging.
   norm=np.sqrt(np.diag(expect)); mflat=flat/norm[:,None]; qdirect=0.
   for distance in range(1,13):
    mix=(lam**distance/4)@mflat
    qdirect+=(16-distance)/16*(mix@mix)
   assert abs(q-qdirect)<1e-10
  rows.append([rid,r['wd'],r['beta'],r['seed'],st,av,rawav,q,*R[upper],*K[upper],*np.sqrt(np.diag(K))])
 tmp=dest.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(cols);w.writerows(rows)
 os.replace(tmp,dest)
 return rid
if __name__=='__main__':
 targets=sorted(manifest,key=lambda k:(manifest[k]['wd'],manifest[k]['beta'],manifest[k]['seed']))
 pilot='--pilot' in sys.argv
 if pilot:targets=['b0.25_wd0.1_s0']
 t=time.time()
 with ProcessPoolExecutor(1 if pilot else 18) as ex:
  for i,r in enumerate(ex.map(run,targets)):
   if i%10==0:print(i+1,r,round(time.time()-t,1),flush=True)
