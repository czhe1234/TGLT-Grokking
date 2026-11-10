"""CPU-only extension of the frozen parameter polar order using original continuation checkpoints."""
import os
for v in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[v]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,importlib.util,hashlib,csv,json,time,io
from pathlib import Path
import torch,numpy as np
torch.set_num_threads(1);sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent
analysis=out.parents[1]
campaign=analysis/'firstgen_beta_wd_gray_continuation_20261102'
work=campaign/'continuation_300k'
spec=importlib.util.spec_from_file_location('rules',out.parent/'coupling_rule_overlaps/measure.py')
rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
formula=rules.formula
with (campaign/'cohort_105.csv').open(encoding='utf-8-sig') as f: roster={r['run_id']:r for r in csv.DictReader(f)}
assert set(roster)==set(rules.manifest)
def folder(r):return Path(r['previous_300k']) if r['previous_300k'] else work/r['run_id']
targets=[k for k,r in roster.items() if r['previous_300k'] or r['status'].startswith('continue_')]
assert len(targets)==78
@torch.inference_mode()
def calc(k):
 p=out/'polar'/f'{k}.csv';r=roster[k];d=folder(r);seed=int(r['seed']);b=float(r['beta']);w=float(r['wd'])
 if p.exists():return k
 rows=[]
 for t in range(100500,300001,500):
  name=f'{t:07d}'; rec=json.loads((d/f'receipt_{name}.json').read_text())
  assert rec['step']==t and rec['complete'] and rec['optimizer_state_saved'] and rec['RNG_saved'] and rec['loader_saved']
  raw=(d/f'checkpoint_{name}.pt').read_bytes()
  assert len(raw)==rec['checkpoint_bytes'] and hashlib.sha256(raw).hexdigest()==rec['checkpoint_sha256']
  cp=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);cfg=cp['config'];sd=cp['model_state']
  assert cp['step']==t and (cfg['seed'],cfg['local_coupling'],cfg['wd'],cfg['p'],cfg['M'],cfg['layers'],cfg['r'],cfg['T'],cfg['seq_len'],cfg['block_type'],cfg['learnable_lambda'])==(seed,b,w,59,4,2,3,4,16,'full',False)
  keys=[key for key in sd if key.endswith('raw_lambda')];assert all(torch.equal(sd[key],sd[keys[0]]) for key in keys)
  lm=(torch.sigmoid(sd[keys[0]].double())*.999).numpy()
  kap=formula.kernel(lm,12,16);fs=formula.factors(sd,1,cfg);K=formula.coupling_gram(fs,kap);R=formula.metrics(K)
  upper=R[np.triu_indices(4,k=1)];q=float(R.sum()/16)
  assert 0<=q<=1+1e-12 and abs(q-(1+3*upper.mean())/4)<1e-12
  if k=='b0.5_wd0.3_s2' and t in (108000,300000):
   om,v,e,gate=fs;flat=np.stack([[gate[m]*np.einsum('ok,ik,jk->oij',om,v[m],e[m,s],optimize=True) for s in range(2)] for m in range(4)]).reshape(4,-1)
   normalized=flat/np.sqrt(np.einsum('ij,ij->i',flat,flat)*np.diag(kap))[:,None]
   direct=0.
   for dist in range(1,13):
    z=(lm**dist/4)@normalized
    direct+=(16-dist)/16*(z@z)
   assert abs(q-direct)<1e-11
  rows.append([k,t,np.sqrt(q),rec['checkpoint_sha256']])
 with p.with_suffix('.tmp').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['run','step','P','checkpoint_sha256']);writer.writerows(rows)
 os.replace(p.with_suffix('.tmp'),p);return k
if __name__=='__main__':
 start=time.time()
 with ProcessPoolExecutor(max_workers=16) as ex:
  jobs=[ex.submit(calc,k) for k in targets]
  for i,f in enumerate(as_completed(jobs)):
   k=f.result()
   if (i+1)%10==0 or i==0 or i+1==len(targets):print(i+1,k,round(time.time()-start,1),flush=True)
