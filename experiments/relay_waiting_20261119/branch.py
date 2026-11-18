"""Paired, short continuations from original H12 states; local record after each evaluation."""
import os
import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO.parent
PKG=ROOT/"releases/TGLT_mechanism_data_20261111"
POLICY=json.loads((HERE/"protocol.json").read_text())
ZU={x["gpu"] for x in POLICY["forks"]}
assert os.environ.get("CUDA_VISIBLE_DEVICES") in ZU
for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import torch
import torch.nn.functional as F
from order import cycles,rotate,P
sys.path.insert(0,str(PKG/"src"))
import exp2_sequence_local_grokking as ref

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
torch.set_default_dtype(torch.float32)
assert torch.cuda.device_count()==1 and torch.__version__=="2.5.1+cu121"
dev=torch.device("cuda:0")
uid=str(torch.cuda.get_device_properties(dev).uuid)
assert (uid if uid.startswith("GPU-") else "GPU-"+uid)==os.environ["CUDA_VISIBLE_DEVICES"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,allow_nan=False,ensure_ascii=False))
    tmp.replace(path)


def get(rid,step):
    with (REPO/"experiments/relay_learning_20261118/results/trajectories.csv").open() as f:
        row=next(x for x in csv.DictReader(f) if x["run_id"]==rid and int(x["step"])==step)
    path=Path(row["path"])
    assert sha(path)==row["sha256"]
    ck=torch.load(path,map_location="cpu",weights_only=False)
    assert ck["step"]==step
    return ck,path,row["sha256"]


def build(cp):
    cfg=ref.RunConfig(**cp["config"])
    assert cfg.batch_size==0 and cfg.split_seed==0 and cfg.local_coupling==.25 and cfg.wd==.03
    assert (cfg.r,cfg.T,cfg.layers,cfg.p,cfg.pos_a,cfg.pos_b,cfg.query_pos)==(3,4,2,59,0,7,15)
    assert cfg.ce_weight==1 and cfg.alpha_field==0 and not (cfg.random_labels or cfg.learnable_lambda or cfg.mode_mask_after or cfg.lambda_override_after)
    kwargs={k:v for k,v in cp["config"].items() if k in inspect.signature(ref.SequenceLocalModeModel).parameters}
    kwargs["lambdas"]=cfg.lambda_values or ref.xi_spread_lambdas(4,cfg.xi_min,cfg.xi_max)
    m=ref.SequenceLocalModeModel(**kwargs).to(dev)
    m.load_state_dict(cp["model_state"])
    opt=ref.make_optimizer(m,cfg)
    opt.load_state_dict(cp["optimizer_state"])
    random.setstate(cp["rng_state"]["python"])
    torch.set_rng_state(cp["rng_state"]["torch"].cpu())
    torch.cuda.set_rng_state(cp["rng_state"]["cuda"][0])
    loader=torch.Generator().set_state(cp["loader_state"])
    assert all(int(opt.state[p]["step"])==cp["step"] for gr in opt.param_groups for p in gr["params"])
    return m,opt,cfg,loader


def data(rid,cfg):
    t,r,y,pos=ref.build_sequence_data(cfg.p,cfg.task,cfg.seq_len,cfg.pos_a,cfg.pos_b,cfg.query_pos)
    tr,te=ref.make_split(len(y),cfg.train_fraction,cfg.split_seed)
    split=json.loads((PKG/"metadata"/rid/"split.json").read_text())
    assert tr.tolist()==split["train_idx"] and te.tolist()==split["test_idx"]
    return [(t[i].to(dev),r[i].to(dev),y[i].to(dev)) for i in (tr,te)],tr.to(dev),te.to(dev),y.to(dev)


def scores(z,y):
    true=z.gather(1,y[:,None])[:,0]
    wrong=z.masked_fill(F.one_hot(y,P).bool(),-torch.inf).amax(1)
    return dict(correct=int((z.argmax(1)==y).sum()),ce=float(F.cross_entropy(z,y,reduction="sum"))/len(y),
                margin=float((true-wrong).mean()))


@torch.no_grad()
def evaluate(m,sets,tr,te,y,cycle,scrambled):
    m.eval()
    bl=m.blocks[1]
    cached=[]
    for t,roles,_ in sets:
        h0=m.residue_features(t)+m.role_features(roles)+m.position[None]
        h=m.blocks[0](h0)
        observed={}
        def keep_right(module,args,out):
            observed["arrived"]=out[0][:,15].contiguous()
        def keep_pre(module,args):
            observed["pre"]=args[0][:,15].contiguous()
        handles=[m.sweep.register_forward_hook(keep_right),bl.norm.register_forward_pre_hook(keep_pre)]
        try:
            h2=bl(h)[:,15]
        finally:
            for hook in handles:
                hook.remove()
        z=m.classifier(m.final_norm(h2))
        cached.append((h[:,15],observed["arrived"],observed["pre"],z))
    inv=torch.cat([tr,te]).argsort()
    h,arr,pre,z=[torch.cat(v)[inv] for v in zip(*cached)]
    b=torch.arange(P*P,device=dev)%P
    assert (h-h[:P][b]).abs().max()<1e-4
    v=torch.einsum("bd,mdo->bmo",h[:P].double(),bl.visible_factor.double())
    v-=v.mean(0)
    ax=arr.double().reshape(P,P,4,8)
    aonly=ax.mean(1)
    inc=(ax-ax.mean(0)).reshape(P*P,4,8)
    f=torch.einsum("nmc,mco->nmo",inc,bl.mode_factor[:,:8].double())
    q=F.linear((bl.local_coupling*bl.mode_logits.double().softmax(0)[None,:,None]*v[b]*f).sum(1),bl.out.weight.double())
    deleted=m.classifier(m.final_norm(bl.norm((pre.double()-q).float())))
    q=q-q.mean(0)
    qmeans=F.one_hot(y,P).double().T@q/P
    metrics=dict(embedding_order=float(cycle.measure(m.residue.weight[1:])),
                 shuffled_order=float(scrambled.measure(m.residue.weight[1:])),
                 received_order=float(cycle.measure(aonly)),local_b_order=float(cycle.measure(v)),
                 query_sum_fraction=float(P*qmeans.square().sum()/q.square().sum().clamp_min(1e-30)))
    for name,ans,ix in (("train",z,tr),("test",z,te),("query_removed",deleted,te)):
        metrics.update({name+"_"+k:value for k,value in scores(ans[ix],y[ix]).items()})
    return metrics


def run(fork,arm):
    rid=fork["run"]
    origin=fork["step"]
    name=fork["name"]+"__"+arm
    dest=HERE/"records"/name
    dest.mkdir(exist_ok=True,parents=True)
    state_dir=HERE/".branch_state"/name
    state_dir.mkdir(exist_ok=True,parents=True)
    cp,path,checksum=get(rid,origin)
    protocol_sha=sha(HERE/"protocol.json")
    meta=dict(name=name,run_id=rid,fork_step=origin,arm=arm,uuid=uid,source=str(path),source_sha256=checksum,
              protocol_sha256=protocol_sha,torch=torch.__version__,source_module_sha256=sha(Path(ref.__file__)),
              code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip())
    assert meta["source_module_sha256"]=="e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73"
    end=POLICY["pulse_steps"]+POLICY["follow_steps"]
    if (dest/"complete.json").exists():
        old=json.loads((dest/"complete.json").read_text())
        assert old["source_sha256"]==checksum and old["protocol_sha256"]==protocol_sha and old["steps"]==end
        return
    ref_hist={int(r["step"]):r for r in json.loads((PKG/"metadata"/rid/"history.json").read_text())}
    resume=state_dir/"latest.pt"
    base=cp
    if resume.exists():
        base=torch.load(resume,map_location="cpu",weights_only=False)
        assert base["meta"]["source_sha256"]==checksum and base["meta"]["protocol_sha256"]==protocol_sha and base["meta"]["name"]==name
    m,opt,cfg,loader=build(base)
    start=base["step"]-origin
    history=base.get("branch_history",[])
    sets,tr,te,y=data(rid,cfg)
    cycles_real,cycles_sham=cycles(dev,POLICY["shuffle_seed"])
    gen0=torch.cuda.get_rng_state().clone()
    check_max=torch.zeros(3,dtype=torch.float64,device=dev)
    impulses=torch.zeros(2,dtype=torch.float64,device=dev)
    train_x,train_roles,train_y=sets[0]
    angle=math.radians(POLICY["angle_degrees"])
    now=time.monotonic()
    def log(rel):
        row=evaluate(m,sets,tr,te,y,cycles_real,cycles_sham)
        row.update(relative_step=rel,step=origin+rel,implementation_norm_error=float(check_max[0]),
                   implementation_radial_error=float(check_max[1]),implementation_mean_error=float(check_max[2]),
                   pulse_order_extra_linear=float(impulses[0]),pulse_native_order_linear=float(impulses[1]))
        if origin+rel in ref_hist:
            row["historical_train_correct"]=ref_hist[origin+rel]["train_correct"]
            row["historical_test_correct"]=ref_hist[origin+rel]["test_correct"]
        if rel==0:
            assert row["train_correct"]==row["historical_train_correct"] and row["test_correct"]==row["historical_test_correct"]
        history.append(row)
        save(dest/"curve.json",history)
        check_max.zero_()
        impulses.zero_()
    if start==0:
        log(0)
    save(dest/"meta.json",meta)
    for rel in range(start+1,end+1):
        m.train()
        for g in opt.param_groups:
            g["lr"]=ref.learning_rate(origin+rel,cfg)
        touching=arm!="native" and rel<=POLICY["pulse_steps"]
        if touching:
            before=m.residue.weight[1:].detach().double()
        opt.zero_grad(set_to_none=True)
        z=m(train_x,train_roles)
        F.cross_entropy(z,train_y).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip,error_if_nonfinite=True)
        opt.step()
        if touching:
            with torch.no_grad():
                weights=m.residue.weight[1:]
                updated,diag=rotate(before,weights,cycles_real,cycles_sham,arm,angle)
                weights.copy_(updated)
                check_max=torch.maximum(check_max,diag[:3])
                impulses+=diag[3:]
        if rel%POLICY["eval_every"]==0:
            log(rel)
        if rel%500==0:
            assert torch.equal(loader.get_state(),cp["loader_state"]) and torch.equal(torch.cuda.get_rng_state(),gen0)
            assert all(int(opt.state[p]["step"])==origin+rel for gr in opt.param_groups for p in gr["params"])
            tmp=state_dir/"resume.tmp"
            payload=dict(config=cp["config"],step=origin+rel,model_state=m.state_dict(),optimizer_state=opt.state_dict(),
                         rng_state=dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=[torch.cuda.get_rng_state()]),
                         loader_state=loader.get_state(),branch_history=history,meta=meta)
            torch.save(payload,tmp)
            tmp.replace(resume)
            if rel==POLICY["pulse_steps"]:
                import shutil
                shutil.copyfile(resume,state_dir/"pulse_end.pt")
            print(name,rel,history[-1]["test_correct"],round(history[-1]["embedding_order"],4),round(time.monotonic()-now,1),flush=True)
    assert sha(path)==checksum and len(history)==end//POLICY["eval_every"]+1
    save(dest/"complete.json",dict(meta,steps=end,elapsed_this_process=time.monotonic()-now,points=len(history)))
    del m,opt


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("fork")
    args=parser.parse_args()
    fork=next(x for x in POLICY["forks"] if x["name"]==args.fork)
    assert fork["gpu"]==os.environ["CUDA_VISIBLE_DEVICES"]
    for arm in POLICY["arms"]:
        run(fork,arm)
