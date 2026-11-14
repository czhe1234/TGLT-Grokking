"""Dispatcher for the 15 authorized runs; idle allowed GPUs can take queued jobs."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl
B=Path(__file__).resolve().parent;PY="/home/chenzhe/.conda/envs/risk-exp4/bin/python"
SW="/home/chenzhe/work/tglt_experiments/.venv-swanlab/bin/python"
GPU=["GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009","GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0","GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"]
def dump(x,p):
 tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def process_matches(pid,rid):
 try:
  cmd=Path("/proc/%d/cmdline"%pid).read_bytes().decode().split("\0")
  seed=rid.rsplit("_s",1)[1]
  window="global" if "global" in rid else str(int(rid.split("_W",1)[1].split("_",1)[0]))
  return (str(B/"train.py") in cmd and cmd[cmd.index("--seed")+1]==seed
          and cmd[cmd.index("--window")+1]==window)
 except (FileNotFoundError,ValueError,IndexError):return False
def main():
 lock=(B/"pipeline.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (B/"pipeline.pid").write_text(str(os.getpid()))
 jobs=[]
 for w in ["8","12","7","15","global"]:
  for s in range(3):
   rid=("transformer_global" if w=="global" else "transformer_W%02d"%int(w))+"_s%d"%s
   jobs.append(dict(window=w,seed=s,preferred_gpu=GPU[s],run_id=rid))
 dump(jobs,B/"jobs.json")
 active={};finished=[];failed=[]
 # Adopt verified in-flight workers without stopping or modifying them.
 for job in jobs:
  path=B/"runs"/job["run_id"]/"status.json"
  if not path.exists():continue
  st=json.loads(path.read_text())
  if st["state"]=="complete" and st["step"]==100000:finished.append(job["run_id"])
  elif st["state"]=="running":
   gpu=st["gpu_uuid"];assert gpu in GPU and gpu not in active
   assert process_matches(st["pid"],job["run_id"]),("Missing in-flight worker",job,st)
   active[gpu]=dict(job,actual_gpu=gpu,pid=st["pid"],child=None,stream=None)
   print(json.dumps(dict(event="adopted",run_id=job["run_id"],pid=st["pid"],gpu=gpu)),flush=True)
  else:raise RuntimeError("Inspect existing failed/incomplete run: "+job["run_id"])
 while len(finished)+len(failed)<len(jobs):
  for gpu,entry in list(active.items()):
   path=B/"runs"/entry["run_id"]/"status.json"
   st=json.loads(path.read_text()) if path.exists() else dict(state="starting",step=0)
   child=entry["child"];rc=child.poll() if child is not None else None
   if st.get("state") in ["complete","failed"] or (child is not None and rc is not None) or (child is None and not process_matches(entry["pid"],entry["run_id"])):
    ok=st.get("state")=="complete" and st.get("step")==100000
    (finished if ok else failed).append(entry["run_id"])
    if entry["stream"] is not None:entry["stream"].close()
    del active[gpu]
    print(json.dumps(dict(event="ended",run_id=entry["run_id"],state=st.get("state"),step=st.get("step"))),flush=True)
  for gpu in GPU:
   if gpu in active:continue
   pending=[j for j in jobs if j["run_id"] not in finished+failed and j["run_id"] not in [v["run_id"] for v in active.values()]]
   if not pending:continue
   job=next((j for j in pending if j["preferred_gpu"]==gpu),pending[0])
   dest=B/"runs"/job["run_id"];dest.mkdir(parents=True,exist_ok=True)
   assert not (dest/"history.json").exists(),"Do not overwrite or repeat a run"
   f=(dest/"train.log").open("a")
   cmd=[PY,str(B/"train.py"),"--window",job["window"],"--seed",str(job["seed"]),"--gpu",gpu]
   child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu))
   active[gpu]=dict(job,actual_gpu=gpu,pid=child.pid,child=child,stream=f)
   lf=(dest/"swanlab.log").open("a")
   subprocess.Popen([SW,str(B/"swanlab_bridge.py"),str(dest)],stdout=lf,stderr=subprocess.STDOUT,
       env=dict(os.environ,CUDA_VISIBLE_DEVICES=""));lf.close()
   print(json.dumps(dict(event="started",run_id=job["run_id"],pid=child.pid,gpu=gpu,command=cmd)),flush=True)
  states=[]
  for entry in active.values():
   path=B/"runs"/entry["run_id"]/"status.json"
   st=json.loads(path.read_text()) if path.exists() else dict(state="starting",step=0)
   states.append(dict(run_id=entry["run_id"],window=entry["window"],seed=entry["seed"],gpu=entry["actual_gpu"],
      **{k:v for k,v in st.items() if k in ["state","step","latest","pid"]}))
  dump(dict(updated=time.time(),running=states,complete=finished,failed=failed,
   queued=[j["run_id"] for j in jobs if j["run_id"] not in finished+failed and j["run_id"] not in [v["run_id"] for v in active.values()]]),B/"status.json")
  if active:time.sleep(5)
 dump(dict(complete=finished,failed=failed,training_complete=not failed,updated=time.time()),B/"training_complete.json")
if __name__=="__main__":main()
