"""Read the historical model by its own forward, editing only the named site."""
import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pairs import measurements

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[2]
PKG = ROOT / "releases/TGLT_mechanism_data_20261111"
sys.path.insert(0, str(PKG / "src"))
import exp2_sequence_local_grokking as ref

A800 = "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"
H100 = "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"
SOURCE_HASH = "e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73"
RUNS = {
    "b0.25_wd0.03_s0": (A800, (4000,107000,155000,300000)),
    "b0.25_wd0.03_s2": (H100, (4000,107000,155000,300000)),
    "r2_T4_M4_seed1": (H100, (3000,51000,58000,100000)),
    "r2_T4_M4_seed0": (H100, (3000,51000,58000,100000)),
}
CONDITIONS = [f"mask_{m:02d}" for m in range(16)] + ["no7_a","no7_joint","no_query","wrong_query_b1"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_at(row):
    path = PKG / row["path"]
    assert digest(path) == digest(Path(row["source_path"])) == row["sha256"]
    cp = torch.load(path, map_location="cpu", weights_only=False)
    cfg = cp["config"]
    assert cp["step"] == int(row["step"]) and cfg["block_type"] == "full"
    assert (cfg["p"], cfg["seq_len"], cfg["pos_a"], cfg["pos_b"], cfg["query_pos"],
            cfg["M"], cfg["layers"], cfg["d"], cfg["d_a"]) == (59,16,0,7,15,4,2,64,8)
    assert not cfg["learnable_lambda"] and not cfg["mode_mask_after"]
    kwargs = {k:v for k,v in cfg.items() if k in inspect.signature(ref.SequenceLocalModeModel).parameters}
    kwargs["lambdas"] = cfg["lambda_values"] or ref.xi_spread_lambdas(4,cfg["xi_min"],cfg["xi_max"])
    model = ref.SequenceLocalModeModel(**kwargs).eval()
    model.load_state_dict(cp["model_state"], strict=True)
    cache = EXPERIMENT / ".native_cache" / f"{row['run_id']}_{cp['step']:07d}.pt"
    after = torch.load(cache, map_location="cpu", weights_only=False)["after"]
    assert {n for n,p in model.named_parameters() if p.requires_grad} == set(after)
    state_after = dict(cp["model_state"])
    state_after.update(after)
    t, r, y, _ = ref.build_sequence_data(cfg["p"],cfg["task"],16,0,7,15)
    tr, te = ref.make_split(len(y),cfg["train_fraction"],cfg["split_seed"])
    indices = json.loads((PKG/"metadata"/row["run_id"]/"split.json").read_text())
    assert tr.tolist() == indices["train_idx"] and te.tolist() == indices["test_idx"]
    ids = torch.arange(3481)
    assert torch.equal(y, (ids//59+ids%59)%59) and (len(tr),len(te)) == (870,2611)
    return model, (cp["model_state"], state_after), tuple(x.cuda() for x in (t,r,y,tr,te)), digest(cache)


def baseline(model, dataset):
    tokens, roles, _, train, test = dataset
    pieces = {}
    def keep(name, val):
        pieces.setdefault(name,[]).append(val.contiguous().clone())
    def input0(_module, args):
        keep("h0", args[0][:,7])
    def input1(_module, args):
        keep("h7",args[0][:,7]); keep("h15",args[0][:,15])
    def observe(_module, args, result):
        calls.append(1)
        right, left = result
        if len(calls)==1:
            keep("source",args[0][:,0]); keep("right7",right[:,7]); keep("left7",left[:,7])
        else:
            keep("right15",right[:,15]); keep("left15",left[:,15])
    def prenorm(_module, args):
        keep("pre7",args[0][:,7])
    assert model.blocks[0].sweep is model.blocks[1].sweep is model.sweep
    hooks = [model.blocks[0].register_forward_pre_hook(input0),
             model.blocks[1].register_forward_pre_hook(input1),
             model.sweep.register_forward_hook(observe),
             model.blocks[0].norm.register_forward_pre_hook(prenorm)]
    try:
        for ix in (train,test):
            calls=[]
            keep("logits",model(tokens[ix], roles[ix]))
            assert len(calls)==2
    finally:
        for h in hooks: h.remove()
    inverse = torch.cat((train,test)).argsort()
    return {key:torch.cat(vals)[inverse] for key,vals in pieces.items()}


def site_edits(model,f):
    first, second = model.blocks
    v = torch.einsum("nd,mdo->nmo", f["h0"], first.visible_factor)
    aux = torch.cat([f["right7"],f["left7"]],-1)
    m = torch.einsum("nmi,mio->nmo", aux,first.mode_factor)
    contributions = first.local_coupling * v*m*first.mode_logits.softmax(0)[None,:,None]
    f["writes"] = F.linear(contributions,first.out.weight)
    pre = f["h0"] + first.out(contributions.sum(1))
    torch.testing.assert_close(pre,f["pre7"],rtol=2e-5,atol=1e-5)
    writes = f["writes"].sum(1).reshape(59,59,-1)
    joint = writes-writes.mean(0,True)-writes.mean(1,True)+writes.mean((0,1),True)
    b = torch.arange(3481,device=pre.device)%59
    h15 = f["h15"].reshape(59,59,-1)
    assert (h15-h15[:1]).abs().max() < (1e-9 if pre.dtype==torch.float64 else 1e-4)
    v = torch.einsum("bd,mdo->bmo",h15[0],second.visible_factor)
    v = v-v.mean(0)
    incoming = torch.cat((f["right15"],f["left15"]),-1).reshape(59,59,4,16)
    incoming=(incoming-incoming.mean(0,True)).reshape(3481,4,16)
    m = torch.einsum("nmi,mio->nmo",incoming,second.mode_factor)
    m = m*second.local_coupling*second.mode_logits.softmax(0)[None,:,None]
    correct = F.linear(m*v[b],second.out.weight).sum(1)
    wrong = F.linear(m*v[(b+1)%59],second.out.weight).sum(1)
    centered = wrong-wrong.mean(1,True)
    scale = (correct-correct.mean(1,True)).norm(dim=1,keepdim=True)/centered.norm(dim=1,keepdim=True).clamp_min(1e-20)
    return dict(no7_a=f["h7"].reshape(59,59,-1).mean(0)[b], no7_joint=-joint.reshape(3481,-1),
                no_query=-correct,wrong_query_b1=-correct+centered*scale)


def direct_forward(model, dataset, condition, edits):
    tokens,roles,_,train,test=dataset
    output=torch.empty(3481,59,device=tokens.device,dtype=model.position.dtype)
    for ix in (train,test):
        count=0
        if condition.startswith("mask"):
            mask=int(condition[5:])
            selected=[j for j in range(4) if (mask>>j)&1]
            def edit(_module, _inputs, result):
                nonlocal count
                count+=1
                if count==1 and selected:
                    right,left = (x.clone() for x in result)
                    right[:,7,selected,:]=0; left[:,7,selected,:]=0
                    return right,left
            hook=model.sweep.register_forward_hook(edit)
        elif condition=="no7_a":
            def replace(_module,_inputs,result):
                nonlocal count
                count+=1
                out=result.clone();out[:,7]=edits[condition][ix]
                return out
            hook=model.blocks[0].register_forward_hook(replace)
        else:
            layer,site=(0,7) if condition=="no7_joint" else (1,15)
            def shift(_module,inputs):
                nonlocal count
                count+=1
                out=inputs[0].clone();out[:,site]+=edits[condition][ix]
                return (out,)
            hook=model.blocks[layer].norm.register_forward_pre_hook(shift)
        try:
            output[ix]=model(tokens[ix],roles[ix])
        finally:
            hook.remove()
        assert count==(2 if condition.startswith("mask") else 1)
    return output


def scores(logits,dataset):
    _,_,y,train,test=dataset
    row={}
    for tag,ix in (("train",train),("test",test)):
        x, labs = logits[ix].double(), y[ix]
        true=x.gather(1,labs[:,None]).squeeze(1)
        wrong=x.masked_fill(F.one_hot(labs,59).bool(),-torch.inf).amax(1)
        count=int((x.argmax(1)==labs).sum())
        row.update({tag+"_correct":count,tag+"_pct":count*100/len(ix),
                    tag+"_ce":float(F.cross_entropy(x,labs)),tag+"_margin":float((true-wrong).mean())})
    return row


def compare_logits(actual,reference,dataset):
    delta=actual.double()-reference
    disagree=actual.argmax(1)!=reference.argmax(1)
    return dict(max_logit_error=float(delta.abs().max()),
                all_predictions_changed=int(disagree.sum()),
                held_predictions_changed=int(disagree[dataset[4]].sum()))


@torch.inference_mode()
def run(row, metadata, prior_modes, prior_deletions):
    rid,step=row["run_id"],int(row["step"])
    model,states,dataset,cache_sha=load_at(row)
    _,_,_,train,test=dataset
    result=dict(run_id=rid,step=step,checkpoint_sha256=row["sha256"],
                source=row["source_path"],after_cache_sha256=cache_sha,terminal_probe=step in (300000,) or (rid.startswith("r2") and step==100000),
                gpu=os.environ["CUDA_VISIBLE_DEVICES"],torch_version=torch.__version__,
                modes=[],offsets=[],deletions=[])
    for stage,weights in zip(("before","after"),states):
        saved=HERE/".work"/f"ref_{rid}_{step}_{stage}.npy"
        reference=torch.from_numpy(np.load(saved)).cuda()
        assert tuple(reference.shape)==(20,3481,59)
        controls={}
        for dtype,prefix in ((torch.float64,"direct64"),(torch.float32,"direct32")):
            model=model.to(device="cuda",dtype=dtype).eval()
            model.load_state_dict(weights,strict=True)
            f=baseline(model,dataset)
            edits=site_edits(model,f)
            if dtype==torch.float64:
                modes,offsets=measurements(model,f,train,test)
                old_modes=prior_modes[(rid,step)] if stage=="before" else metadata["mode_after"]
                for val,old in zip(modes,old_modes):
                    assert val["mode"]==int(old["mode"])
                    val["previous_transport_adj"]=float(old["received_energy"])
                    val["previous_mixed_adj"]=float(old["mixed_difference_energy"])
                    val["previous_read_correct"]=int(old["received_id"])
                    val["stage"]=stage
                result["modes"].extend(modes)
                result["offsets"].extend(dict(stage=stage,**x) for x in offsets)
            for i,condition in enumerate(CONDITIONS):
                z=f["logits"] if i==0 else direct_forward(model,dataset,condition,edits)
                old_z=reference[i]
                if prefix=="direct64":
                    old_stats=scores(old_z,dataset)
                    controls[condition]=dict(stage=stage,condition=condition,
                        **{"prev_"+k:v for k,v in old_stats.items()})
                    if i<16:
                        entry=(prior_deletions[(rid,step)] if stage=="before" else metadata["deletion_after"])[i]
                        assert int(entry["mask"])==i
                        for k in ("train_correct","test_correct","train_ce","test_ce","train_margin","test_margin"):
                            if not np.isclose(float(old_stats[k]),float(entry[k]),rtol=1e-10,atol=1e-8):
                                raise ValueError((rid,step,stage,condition,k,old_stats[k],entry[k]))
                stats=scores(z,dataset)
                if i==0 and prefix=="direct32":
                    expected=metadata["native_"+stage]
                    assert all(stats[t+"_correct"]==expected[t]["correct"] for t in ("train","test")), (rid,step,stage,stats,expected)
                stats.update(compare_logits(z,old_z,dataset))
                controls[condition].update({prefix+"_"+k:v for k,v in stats.items()})
            del f,edits
        result["deletions"].extend(controls.values())
        del reference
    tmp=HERE/".work"/f"{rid}_{step}.tmp"
    tmp.write_text(json.dumps(result,allow_nan=False))
    tmp.replace(tmp.with_suffix(".json"))
    print("verified",rid,step,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="+",choices=RUNS)
    parser.add_argument("--step",type=int)
    opts=parser.parse_args()
    device=os.environ["CUDA_VISIBLE_DEVICES"]
    assert torch.cuda.device_count()==1 and torch.__version__=="2.5.1+cu121"
    assert device=="GPU-"+str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    assert all(RUNS[r][0]==device for r in opts.runs)
    assert digest(Path(ref.__file__))==SOURCE_HASH
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    with (PKG/"checkpoints_manifest.csv").open(encoding="utf-8-sig") as f:
        manifest={(v["run_id"],int(v["step"])):v for v in csv.DictReader(f)}
    def old_table(name,idx):
        table={}
        with (EXPERIMENT/"results"/name).open() as f:
            for v in csv.DictReader(f):
                table.setdefault((v["run_id"],int(v["step"])),[]).append(v)
        return {k:sorted(v,key=lambda r:int(r[idx])) for k,v in table.items()}
    modes=old_table("modes_before.csv","mode");deletions=old_table("delete_before.csv","mask")
    meta={(r["run_id"],r["step"]):r for r in json.loads((EXPERIMENT/"results/native_effects.json").read_text())}
    for rid in opts.runs:
        steps=RUNS[rid][1]
        if opts.step is not None:
            assert opts.step in steps
            steps=(opts.step,)
        for step in steps:
            if not (HERE/".work"/f"{rid}_{step}.json").exists():
                run(manifest[rid,step],meta[rid,step],modes,deletions)


if __name__=="__main__":
    main()
