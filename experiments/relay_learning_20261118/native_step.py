"""Replay one full-batch FP32 update in the original runtime on the verified historical GPU."""
import csv
import hashlib
import inspect
import json
import os
import random
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PKG=ROOT/"releases/TGLT_mechanism_data_20261111"
A800="GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"
H100="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
uid=os.environ["CUDA_VISIBLE_DEVICES"]
assert uid in {A800,H100}
import torch
import torch.nn.functional as F
sys.path.insert(0,str(PKG/"src"))
import exp2_sequence_local_grokking as ref
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
assert torch.cuda.device_count()==1 and torch.__version__=="2.5.1+cu121"
actual=str(torch.cuda.get_device_properties(0).uuid)
assert (actual if actual.startswith("GPU-") else "GPU-"+actual)==uid

def calc(rid,step,record,forward_only=False):
    path=PKG/record["path"]
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==record["sha256"]
    cp=torch.load(path,map_location="cpu",weights_only=False)
    cfg=ref.RunConfig(**cp["config"])
    kwargs={k:v for k,v in cp["config"].items() if k in inspect.signature(ref.SequenceLocalModeModel).parameters}
    kwargs["lambdas"]=cfg.lambda_values or ref.xi_spread_lambdas(4,cfg.xi_min,cfg.xi_max)
    m=ref.SequenceLocalModeModel(**kwargs).to("cuda")
    m.load_state_dict(cp["model_state"])
    opt=ref.make_optimizer(m,cfg)
    opt.load_state_dict(cp["optimizer_state"])
    for g in opt.param_groups:
        g["lr"]=ref.learning_rate(step+1,cfg)
    t,r,y,_=ref.build_sequence_data(cfg.p,cfg.task,cfg.seq_len,cfg.pos_a,cfg.pos_b,cfg.query_pos)
    tr,te=ref.make_split(len(y),cfg.train_fraction,cfg.split_seed)
    split=json.loads((PKG/"metadata"/rid/"split.json").read_text())
    assert tr.tolist()==split["train_idx"] and te.tolist()==split["test_idx"]
    sets=[(t[i].cuda(),r[i].cuda(),y[i].cuda()) for i in [tr,te]]
    @torch.no_grad()
    def evaluate():
        out={}
        m.eval()
        for name,(tokens,roles,label) in zip(["train","test"],sets):
            z=m(tokens,roles)
            true=z.gather(1,label[:,None]).squeeze(1)
            wrong=z.masked_fill(F.one_hot(label,59).bool(),-torch.inf).amax(1)
            out[name]=dict(correct=int(z.argmax(1).eq(label).sum()),
                ce=float(F.cross_entropy(z,label,reduction="sum"))/len(label),margin=float((true-wrong).mean()))
        return out
    before=evaluate()
    last=next(x for x in cp["history"] if int(x["step"])==step)
    assert all(before[tag]["correct"]==last[tag+"_correct"] for tag in ["train","test"]),(rid,step,before,last)
    if forward_only:
        return dict(run_id=rid,step=step,device=uid,before=before,
                    original_test_correct=last["test_correct"],test_ce_abs_error=abs(before["test"]["ce"]-last["test_ce"]))
    random.setstate(cp["rng_state"]["python"])
    torch.set_rng_state(cp["rng_state"]["torch"])
    torch.cuda.set_rng_state(cp["rng_state"]["cuda"][0])
    loader=torch.Generator().set_state(cp["loader_state"])
    rng_before=torch.cuda.get_rng_state().clone()
    m.train()
    opt.zero_grad(set_to_none=True)
    tokens,roles,label=sets[0]
    F.cross_entropy(m(tokens,roles),label).backward()
    gradnorm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip,error_if_nonfinite=True))
    names={id(p):n for n,p in m.named_parameters() if p.requires_grad}
    decayed,grad={},{}
    for gr in opt.param_groups:
        for p in gr["params"]:
            name=names[id(p)]
            decayed[name]=(p.detach()*(1-gr["lr"]*gr["weight_decay"])).cpu()
            grad[name]=p.grad.detach().cpu().clone()
    opt.step()
    after=evaluate()
    assert torch.equal(torch.cuda.get_rng_state(),rng_before) and torch.equal(loader.get_state(),cp["loader_state"])
    out=dict(run_id=rid,step=step,device=uid,torch_version=torch.__version__,gradnorm=gradnorm,
        clip_factor=min(1.,cfg.grad_clip/(gradnorm+1e-6)),before=before,after=after,
        horizon_plus_one=(step==cfg.steps),sha256=sha)
    if step==0:
        hist=json.loads((PKG/"metadata"/rid/"history.json").read_text())
        true=next(x for x in hist if x["step"]==1)
        for name in ["train","test"]:
            assert after[name]["correct"]==true[name+"_correct"]
            assert abs(after[name]["ce"]-true[name+"_ce"])<5e-6,(rid,after[name],true[name+"_ce"])
        out["first_historical_step_verified"]=True
    else:
        cache=HERE/".native_cache"/(rid+"_%07d.pt"%step)
        torch.save(dict(grad=grad,decayed=decayed,after={n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad}),cache)
    assert sha==hashlib.sha256(path.read_bytes()).hexdigest()
    print(rid,step,gradnorm,before["test"]["correct"],after["test"]["correct"],flush=True)
    return out


if __name__=="__main__":
    runs=("b0.25_wd0.03_s0",) if uid==A800 else ("b0.25_wd0.03_s2","r2_T4_M4_seed1","r2_T4_M4_seed0")
    (HERE/".native_cache").mkdir(exist_ok=True)
    with (PKG/"checkpoints_manifest.csv").open(encoding="utf-8-sig") as f:
        manifest={(r["run_id"],int(r["step"])):r for r in csv.DictReader(f)}
    if "--verify-counts" in sys.argv:
        result=[]
        with (HERE/"results/trajectories.csv").open() as f:
            for r in csv.DictReader(f):
                if r["run_id"] in runs and (int(r["replay_delta_test"]) or int(r["replay_delta_train"])):
                    result.append(calc(r["run_id"],int(r["step"]),r,True))
        suffix="A800" if uid==A800 else "H100"
        (HERE/"results"/("precision_"+suffix+".json")).write_text(json.dumps(result,indent=2))
        sys.exit(0)
    result=[]
    for rid in runs:
        for t in ([0,4000,107000,155000,300000] if rid.startswith("b") else [0,3000,51000,58000,100000]):
            result.append(calc(rid,t,manifest[rid,t]))
    name="native_A800.json" if uid==A800 else "native_H100.json"
    (HERE/"results"/name).write_text(json.dumps(result,indent=2))
