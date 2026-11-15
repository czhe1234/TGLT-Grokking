"""Run exactly 18 authorized controls, at most two workers on each allowed GPU."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl
B=Path(__file__).resolve().parent;PY='/home/chenzhe/.conda/envs/risk-exp4/bin/python'
SW=str(B.parents[1]/'.venv-swanlab/bin/python')
GPU=json.loads((B/'config.json').read_text())['allow_gpu_uuids'];CAP=2

def dump(x,p):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def alive(entry):
 try:
  cmd=Path('/proc/%d/cmdline'%entry['pid']).read_bytes().decode().split('\0')
  return str(B/'train.py') in cmd and cmd[cmd.index('--group')+1]==entry['group'] and cmd[cmd.index('--window')+1]==str(entry['window']) and cmd[cmd.index('--seed')+1]==str(entry['seed'])
 except (OSError,ValueError,IndexError):return False

def main():
 lock=(B/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (B/'pipeline.pid').write_text(str(os.getpid()));jobs=[]
 for w in [8,12,15]:
  for group in ['T1','T2']:
   for seed in range(3):jobs.append(dict(group=group,window=w,seed=seed,preferred_gpu=GPU[seed],run_id='%s_W%02d_s%d'%(group,w,seed),steps=150000 if group=='T1' else 100000))
 dump(jobs,B/'jobs.json');active={};finished=[];failed=[]
 for job in jobs:
  sp=B/'runs'/job['run_id']/'status.json'
  if not sp.exists():continue
  st=json.loads(sp.read_text())
  if st['state']=='complete' and st['step']==job['steps']:finished.append(job['run_id'])
  elif st['state']=='running':
   entry=dict(job,gpu_uuid=st['gpu_uuid'],pid=st['pid'],child=None,stream=None)
   assert entry['gpu_uuid'] in GPU and alive(entry);active[job['run_id']]=entry
  else:raise RuntimeError('Inspect incomplete/failed run '+job['run_id'])
 while len(finished)+len(failed)<len(jobs):
  for rid,entry in list(active.items()):
   sp=B/'runs'/rid/'status.json';st=json.loads(sp.read_text()) if sp.exists() else dict(state='starting',step=0)
   child=entry['child'];exited=child.poll() is not None if child is not None else not alive(entry)
   if st.get('state') in ['complete','failed'] or exited:
    ok=st.get('state')=='complete' and st.get('step')==entry['steps']
    (finished if ok else failed).append(rid)
    if entry['stream'] is not None:entry['stream'].close()
    del active[rid];print(json.dumps(dict(event='ended',run_id=rid,state=st.get('state'),step=st.get('step'))),flush=True)
  for gpu in GPU:
   while sum(v['gpu_uuid']==gpu for v in active.values())<CAP:
    pending=[j for j in jobs if j['run_id'] not in finished+failed and j['run_id'] not in active]
    if not pending:break
    job=next((j for j in pending if j['preferred_gpu']==gpu),pending[0]);dest=B/'runs'/job['run_id'];dest.mkdir(exist_ok=True)
    assert not (dest/'history.json').exists(),'Refuse overwrite'
    f=(dest/'train.log').open('a');cmd=[PY,str(B/'train.py'),'--group',job['group'],'--window',str(job['window']),'--seed',str(job['seed']),'--gpu',gpu]
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu))
    active[job['run_id']]=dict(job,gpu_uuid=gpu,pid=child.pid,child=child,stream=f)
    lf=(dest/'swanlab.log').open('a');subprocess.Popen([SW,str(B/'swanlab_bridge.py'),str(dest)],stdout=lf,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));lf.close()
    print(json.dumps(dict(event='started',run_id=job['run_id'],pid=child.pid,gpu=gpu,command=cmd)),flush=True)
  states=[]
  for rid,entry in active.items():
   sp=B/'runs'/rid/'status.json';st=json.loads(sp.read_text()) if sp.exists() else dict(state='starting',step=0)
   states.append(dict(run_id=rid,group=entry['group'],window=entry['window'],seed=entry['seed'],gpu=entry['gpu_uuid'],target_step=entry['steps'],**{k:v for k,v in st.items() if k in ['state','step','latest','pid']}))
  dump(dict(updated=time.time(),running=states,complete=finished,failed=failed,queued=[j['run_id'] for j in jobs if j['run_id'] not in finished+failed and j['run_id'] not in active]),B/'status.json')
  if active:time.sleep(5)
 dump(dict(complete=finished,failed=failed,training_complete=not failed,updated=time.time()),B/'training_complete.json')
if __name__=='__main__':main()
