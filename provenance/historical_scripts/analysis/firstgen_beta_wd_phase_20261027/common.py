import os,sys,json,hashlib,importlib.util,math,ast
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
SOURCE=ROOT/"source/exp2_sequence_local_grokking.py"
EXPECTED="e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_name(p.name+".tmp")
 t.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False));os.replace(t,p)
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,(bool,np.bool_)):return bool(x)
 if isinstance(x,(int,np.integer)):return int(x)
 return x
def loadmod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
assert sha(SOURCE)==EXPECTED
native=loadmod("phase_historical_native",SOURCE)
g11=loadmod("phase_g11_metrics",ROOT/"source/g11_science_utils.py")
g11.ROOT=ROOT;g11.PROJECT=PROJECT;g11.SOURCE=SOURCE;g11.MOD=native
tree=ast.parse((ROOT/"source/strict_label_source.py").read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="audit")
ns={"math":math};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/"source/strict_label_source.py"),"exec"),ns)
strict_audit=ns["audit"]
def full_data(cfg,device):
 t,r,y,_=native.build_sequence_data(cfg["p"],cfg["task"],cfg["seq_len"],cfg["pos_a"],cfg["pos_b"],cfg["query_pos"],False,cfg["random_label_seed"])
 tr,te=native.make_split(len(y),cfg["train_fraction"],cfg["split_seed"])
 old=read(PROJECT/"week/v1/results"/("r3_T4_M4_seed%d"%cfg["seed"])/"split.json")
 assert tr.tolist()==old["train_idx"] and te.tolist()==old["test_idx"]
 return tuple(x.to(device) for x in (t,r,y,tr,te))
def anchor_check(payload,cfg):
 step=int(payload["step"]);rid="r3_T4_M4_seed%d"%cfg["seed"]
 path=PROJECT/"week/v1/results"/rid/("checkpoint_%07d.pt"%step)
 if not path.exists():return None
 old=torch.load(path,map_location="cpu",weights_only=False)
 a=payload["model_state"];b=old["model_state"]
 delta=max(float((a[n].detach().cpu()-b[n]).abs().max()) for n in b)
 if step==0:g11.exact(a,b,"initial anchor model");g11.exact(payload["rng_state"],old["rng_state"],"initial anchor RNG")
 exact=delta==0
 if exact:
  g11.exact(payload["optimizer_state"],old["optimizer_state"],"anchor optimizer")
 return dict(step=step,model_exact=exact,max_parameter_error=delta,old_checkpoint_sha256=sha(path))
