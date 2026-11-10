import os,sys,argparse
p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);a=p.parse_args()
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
job=next(j for j in json.loads((ROOT/"jobs.json").read_text()) if j["run_id"]==a.run_id)
gpu=job["gpu"]
os.environ["CUDA_VISIBLE_DEVICES"]=gpu["uuid"];os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_MPS_PIPE_DIRECTORY"]=gpu["mps_pipe"]
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
sys.dont_write_bytecode=True
import time,fcntl,traceback,platform
import torch,numpy as np
from common import *
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
assert torch.cuda.device_count()==1
prop=torch.cuda.get_device_properties(0)
actual=str(prop.uuid);actual=actual if actual.startswith("GPU-") else "GPU-"+actual
assert actual==gpu["uuid"]
cfg=read(job["config"]);dest=Path(cfg["out_dir"])
dest.mkdir(exist_ok=True)
lock=(dest/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (dest/"terminal.json").exists() and not list(dest.glob("checkpoint_*.pt")),"Refuse duplicate run"
for k in ("lambda_values","lambda_override_after","mode_mask_after"):cfg[k]=tuple(cfg[k])
inputs=full_data(cfg,"cuda")
start=time.time();step=0;latest={};pending=None;modelref=None;anchor_checks=[]
identity=dict(gpu=gpu,actual_uuid=actual,gpu_name=prop.name,torch=torch.__version__,python=platform.python_version(),source_sha256=EXPECTED,code_hashes={n:sha(ROOT/n) for n in ("worker.py","common.py","protocol.json","source/g11_science_utils.py")},no_hidden_noise=True)
save(dest/"origin.json",dict(**identity,config=cfg,started=time.time(),pid=os.getpid(),historical_initialization_seed=cfg["seed"]))
orig_eval=native.evaluate;orig_csv=native.write_csv;orig_atomic=native.atomic_torch_save
def evaluate(model,*args,**kwargs):
 global modelref
 modelref=model
 return orig_eval(model,*args,**kwargs)
def measure(model):
 before=native.rng_state()
 sm,ar,ck=g11.measure(model,cfg,*inputs,include_mixed=False)
 g11.exact(native.rng_state(),before,"measure does not consume RNG")
 assert max(ck.values())<5e-4,ck
 vals={"D":sm["effective_D"],"I":sm["effective_I"],"rho":sm["effective_rho"],
       "D_coherent":sm["effective_coherent"],"raw_D":sm["raw_D"],"raw_I":sm["raw_I"],
       "LN2_scale_mean":sm["LN2_scale_mean"],"D_mode_0":float(ar["effective_d"][0]),"D_mode_1":float(ar["effective_d"][1]),
       "D_mode_2":float(ar["effective_d"][2]),"D_mode_3":float(ar["effective_d"][3])}
 return vals,ar,ck
def observe(model,*args,**kwargs):
 global pending
 pending=measure(model)
 return pending[0]
def status(state):
 save(dest/"status.json",dict(state=state,run_id=a.run_id,pid=os.getpid(),step=step,total_steps=100000,
      elapsed_seconds=time.time()-start,updates_per_second=step/max(time.time()-start,1e-9),gpu=gpu,
      latest=latest,time=time.time(),strict_so_far=strict_audit(history_current) if history_current else None))
history_current=[]
def csvwrite(history,path):
 global pending,step,latest,history_current
 step=int(history[-1]["step"]);history_current=history
 if step==0 and pending is None:
  pending=measure(modelref);history[-1].update(pending[0])
 if pending is not None:
  vals,arrays,checks=pending
  assert abs(history[-1]["D"]-vals["D"])<1e-12
  g11.save_npz(dest/("rule_fields_%07d.npz"%step),**{k:v for k,v in arrays.items() if k in ["effective_d","effective_S","effective_C","raw_d","raw_S","train_margin","test_margin"]})
  save(dest/("observation_%07d.json"%step),dict(step=step,values=vals,checks=checks));pending=None
 latest={k:history[-1][k] for k in ["train_acc","test_acc","train_ce","test_ce","D","I"] if k in history[-1]}
 orig_csv(history,path);status("running")
def atomic(payload,path):
 global anchor_checks
 assert not Path(path).exists(),"Refuse overwrite"
 if job["anchor"] and int(payload["step"])%1000==0:
  cc=anchor_check(payload,cfg);anchor_checks.append(cc);save(dest/"anchor_parameter_checks.json",anchor_checks)
  if payload["step"]==1000 and cc["max_parameter_error"]>1e-4:
   raise RuntimeError("Anchor prefix mismatch; pause for diagnosis")
 orig_atomic(payload,path)
 save(dest/("receipt_%07d.json"%payload["step"]),dict(step=payload["step"],checkpoint_sha256=sha(path),checkpoint_bytes=Path(path).stat().st_size))
native.evaluate=evaluate;native.observe=observe;native.write_csv=csvwrite;native.atomic_torch_save=atomic;native.plot_history=lambda *a,**k:None
try:
 result=native.train_one_run(native.RunConfig(**cfg))
 h=read(dest/"history.json");label=strict_audit(h)
 assert label["final_step"]==100000
 save(dest/"strict_label.json",label)
 assert read(dest/"split.json")==read(PROJECT/"week/v1/results"/("r3_T4_M4_seed%d"%cfg["seed"])/"split.json")
 save(dest/"terminal.json",dict(state="complete",run_id=a.run_id,beta=job["beta"],wd=job["wd"],seed=job["seed"],strict=label,seconds=time.time()-start,checkpoint_count=len(list(dest.glob("checkpoint_*.pt")))))
 status("complete")
except BaseException:
 save(dest/"terminal.json",dict(state="failed",run_id=a.run_id,step=step,error=traceback.format_exc(),seconds=time.time()-start))
 status("failed");raise
