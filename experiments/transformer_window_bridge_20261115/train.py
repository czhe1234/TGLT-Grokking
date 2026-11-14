"""Train one predeclared window condition. Full checkpoints every 500 steps."""
import os,argparse
p=argparse.ArgumentParser()
p.add_argument("--window",required=True,choices=["7","8","12","15","global"])
p.add_argument("--seed",required=True,type=int,choices=[0,1,2])
p.add_argument("--gpu",required=True)
p.add_argument("--root",default=str(__import__("pathlib").Path(__file__).resolve().parent))
p.add_argument("--pilot",action="store_true")
p.add_argument("--resume",action="store_true")
a=p.parse_args()
ALLOW=["GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009","GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0","GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"]
assert a.gpu in ALLOW
os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
from pathlib import Path
import sys,time,json,random,copy,fcntl,traceback,ctypes,uuid,platform
B=Path(a.root).resolve()
sys.path.insert(0,str(B/"source"))
from exp2_window_transformer import *
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.mha.set_fastpath_enabled(False)
assert torch.cuda.device_count()==1
class CUuuid(ctypes.Structure):_fields_=[("bytes",ctypes.c_ubyte*16)]
drv=ctypes.CDLL("libcuda.so.1");assert drv.cuInit(0)==0
dev=ctypes.c_int();assert drv.cuDeviceGet(ctypes.byref(dev),0)==0
u=CUuuid();assert drv.cuDeviceGetUuid_v2(ctypes.byref(u),dev)==0
actual="GPU-"+str(uuid.UUID(bytes=bytes(u.bytes)));assert actual==a.gpu
assert json.loads((B/"validation/cpu_checks.json").read_text())["passed"]
W=None if a.window=="global" else int(a.window)
rid=("transformer_global" if W is None else "transformer_W%02d"%W)+"_s%d"%a.seed
dest=B/("validation" if a.pilot else "runs")/(("pilot_" if a.pilot else "")+rid)
dest.mkdir(exist_ok=True)
lock=(dest/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert a.resume or not (dest/"history.json").exists(),"Refusing overwrite"
cfg=dict(json.loads((B/"config.json").read_text()),window=W,seed=a.seed,run_id=rid,
         exp_group="transformer_global" if W is None else "transformer_W%02d"%W)
json_write(cfg,dest/"config.json")
d=make_data();t,r,y=[d[k].cuda() for k in ["tokens","roles","labels"]];tr,te=d["train"].cuda(),d["test"].cuda()
tt,rr,yy=t[tr],r[tr],y[tr]
model=WindowTransformer(a.seed,W).cuda();op=optimizer(model,True)
history=[];step=0;start=time.time();elapsed_previous=0.
origin=dict(pid=os.getpid(),gpu_uuid=actual,gpu_name=torch.cuda.get_device_name(),
    python=platform.python_version(),torch=torch.__version__,parameter_count=ref.count_params(model),
    initial_parameter_hash=param_hash(model),data_sha256=sha(B/"data.pt"),
    code_hashes={x:sha(B/x) for x in ["train.py","source/exp2_window_transformer.py","source/strict_grokking.py","source/exp2_sequence_local_grokking.py","config.json"]})
if not a.resume:json_write(origin,dest/"origin.json")
if a.resume:
    cp=torch.load(sorted(dest.glob("checkpoint_*.pt"))[-1],map_location="cuda",weights_only=False)
    assert cp["config"]==cfg and cp["data_sha256"]==sha(B/"data.pt")
    model.load_state_dict(cp["model_state"]);op.load_state_dict(cp["optimizer_state"])
    history=cp["history"];step=cp["step"];elapsed_previous=history[-1]["wall_sec"]
    random.setstate(cp["rng_state"]["python"]);torch.set_rng_state(cp["rng_state"]["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in cp["rng_state"]["cuda"]])
    for group in op.param_groups:group["lr"]=torch.tensor(float(group["lr"]),device="cuda")
    json_write(dict(pid=os.getpid(),step=step,gpu_uuid=actual),dest/"resume.json")
# CUDA graph warmup is rolled back. It does not add training steps.
saved_model={k:v.detach().clone() for k,v in model.state_dict().items()}
saved_optimizer=copy.deepcopy(op.state_dict());initial_hash=param_hash(model)
def train_step():
    op.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model(tt,rr),yy);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True,error_if_nonfinite=False)
    op.step()
    return loss,norm
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):train_step()
torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):graph_loss,graph_norm=train_step()
model.load_state_dict(saved_model)
flat=[x for g in op.param_groups for x in g["params"]]
saved_ids=[x for g in saved_optimizer["param_groups"] for x in g["params"]]
for param,pid in zip(flat,saved_ids):
    state=saved_optimizer["state"].get(pid)
    for key,v in op.state[param].items():
        if isinstance(v,torch.Tensor):v.copy_(state[key] if state is not None else torch.zeros_like(v))
torch.cuda.synchronize();assert param_hash(model)==initial_hash
del saved_model,saved_optimizer

@torch.no_grad()
def savepoint():
    model.eval()
    z=torch.cat([model(t[i:i+512],r[i:i+512]) for i in range(0,len(t),512)])
    correct=z.argmax(-1)==y;ce=F.cross_entropy(z,y,reduction="none")
    margin=z.gather(1,y[:,None])[:,0]-z.scatter(1,y[:,None],-torch.inf).max(-1).values
    row=dict(step=step,wall_sec=elapsed_previous+time.time()-start,lr=float(op.param_groups[0]["lr"]),
             weight_norm=sum(p.square().sum().item() for p in model.parameters())**.5)
    for name,ix in [("train",tr),("test",te)]:
        row.update({name+"_acc":int(correct[ix].sum())/len(ix),name+"_correct":int(correct[ix].sum()),
            name+"_n":len(ix),name+"_ce":float(ce[ix].mean()),name+"_margin_mean":float(margin[ix].mean()),
            name+"_margin_median":float(margin[ix].median())})
    assert all(__import__("math").isfinite(v) for v in row.values())
    history.append(row);json_write(history,dest/"history.json");ref.write_csv(history,dest/"history.csv")
    if not a.pilot:
        ref.atomic_torch_save(dict(step=step,model_state=model.state_dict(),optimizer_state=op.state_dict(),
            rng_state=ref.rng_state(),config=cfg,history=history,
            train_idx=d["train"],test_idx=d["test"],data_sha256=sha(B/"data.pt")),
            dest/("checkpoint_%07d.pt"%step))
    json_write(dict(state="running",step=step,pid=os.getpid(),gpu_uuid=actual,updated=time.time(),latest=row),dest/"status.json")
    print(json.dumps(row),flush=True)
    model.train()
try:
    if not history:savepoint()
    limit=500 if a.pilot else 100000
    bench=time.monotonic();bench_start=step
    for step in range(step+1,limit+1):
        if step<=101:
            for group in op.param_groups:group["lr"].fill_(.001*min(1.,step/100.))
        graph.replay()
        if a.pilot and step==100:
            torch.cuda.synchronize();bench=time.monotonic();bench_start=100
        if step%500==0:
            torch.cuda.synchronize()
            assert bool(torch.isfinite(graph_loss)) and bool(torch.isfinite(graph_norm))
            savepoint()
    torch.cuda.synchronize()
    if a.pilot:
        assert history[-1]["train_ce"]<history[0]["train_ce"]
        json_write(dict(passed=True,seconds_per_1000=(time.monotonic()-bench)/(step-bench_start)*1000,
            peak_memory=torch.cuda.max_memory_allocated(),initial=history[0],final=history[-1]),dest/"pilot.json")
    else:
        result=classify(history);result.update(run_id=rid,window=W,seed=a.seed,parameter_count=ref.count_params(model))
        json_write(result,dest/"summary.json")
    json_write(dict(state="complete",step=step,pid=os.getpid(),gpu_uuid=actual,updated=time.time(),latest=history[-1]),dest/"status.json")
except BaseException:
    json_write(dict(state="failed",step=step,pid=os.getpid(),gpu_uuid=actual,updated=time.time(),error=traceback.format_exc()),dest/"status.json")
    raise
