"""Original mod-59 checkpoints, native layers and local interventions."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import hashlib
import inspect
import json
import runpy
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path(os.environ.get("TGLT_SOURCE_ROOT", REPO.parent))
os.environ["TGLT_SOURCE_ROOT"] = str(ROOT)
PACKAGE = ROOT / "releases/TGLT_mechanism_data_20261111"
OUT = HERE / "results"
training = runpy.run_path(str(REPO / "experiments/training_structure_20261117/analyze.py"))
bridge = training["bridge"]
ref = bridge.ref
P = 59
N = P * P
D = 64
M = 4
I = torch.arange(N)
A, B = I // P, I % P
DONOR = ((A + 1) % P) * P + B
DATA = bridge.make_data()
Y, TR, TE = DATA["labels"], DATA["train"], DATA["test"]
HELD = torch.zeros(N, dtype=torch.bool)
HELD[TE] = True
assert torch.equal(Y, (A + B) % P) and len(TE) == 2611 and len(TR) == 870
RUNS = ("b0.25_wd0.03_s0", "b0.25_wd0.03_s2", "r2_T4_M4_seed1", "r2_T4_M4_seed0")
ANCHORS = ((4000, 107000, 155000, 300000), (3000, 51000, 58000, 100000))
SRC_HASH = "e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


def table(path, rows):
    fields=sorted(set().union(*(row.keys() for row in rows)))
    temporary=path.with_suffix(".tmp")
    with temporary.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@lru_cache(maxsize=1)
def manifest():
    with (PACKAGE / "checkpoints_manifest.csv").open(encoding="utf-8-sig") as f:
        return {(r["run_id"], int(r["step"])):r for r in csv.DictReader(f)}


@lru_cache(maxsize=4)
def metadata(rid):
    base = PACKAGE / "metadata" / rid
    split = read_json(base / "split.json")
    assert split["train_idx"] == TR.tolist() and split["test_idx"] == TE.tolist()
    assert digest(Path(ref.__file__)) == SRC_HASH
    return {int(r["step"]):r for r in read_json(base / "history.json")}


def anchor(rid, step):
    return step in ANCHORS[int(rid.startswith("r2"))]


def load(rid, step):
    filename = "checkpoint_%07d.pt" % step
    record = manifest().get((rid, step))
    if record:
        path = PACKAGE / record["path"]
        checksum = digest(path)
        assert checksum == record["sha256"] == digest(Path(record["source_path"]))
    else:
        if rid.startswith("r2"):
            runroot = ROOT/"week/v1/results"
        elif step <= 100000:
            runroot = ROOT/"analysis/firstgen_beta_wd_phase_20261027/runs"
        else:
            runroot = ROOT/"analysis/firstgen_beta_wd_gray_continuation_20261102/continuation_300k"
        path = runroot/rid/filename
        checksum = digest(path)
    cp = torch.load(path, map_location="cpu", weights_only=False)
    cfg = cp["config"]
    assert {"optimizer_state","model_state","loader_state","rng_state","history","step"} <= cp.keys()
    assert cp["step"] == step and cfg["batch_size"] == 0 and cfg["ce_weight"] == 1 and cfg["alpha_field"] == 0
    assert not (cfg["learnable_lambda"] or cfg["mode_mask_after"] or cfg["lambda_override_after"] or cfg["random_labels"])
    assert (cfg["p"],cfg["pos_a"],cfg["pos_b"],cfg["query_pos"],cfg["M"],cfg["layers"]) == (59,0,7,15,4,2)
    assert (cfg["local_coupling"],cfg["wd"],cfg["r"]*cfg["T"]) == ((1.,.1,8) if rid.startswith("r2") else (.25,.03,12))
    kwargs = {k:v for k,v in cfg.items() if k in inspect.signature(ref.SequenceLocalModeModel).parameters}
    kwargs["lambdas"] = cfg["lambda_values"] or ref.xi_spread_lambdas(4,cfg["xi_min"],cfg["xi_max"])
    model = ref.SequenceLocalModeModel(**kwargs).double().eval()
    model.load_state_dict(cp["model_state"])
    native = model.sweep
    sweep = bridge.MatrixSweep(4, cfg["r"], cfg["T"], native.lambdas.tolist(), False).double()
    sweep.load_state_dict(native.state_dict())
    sweep.rebuild()
    model.sweep = sweep
    for block in model.blocks:
        block.sweep = sweep
    return model, cp, native, str(path), checksum


def score(logits, ix=TE, labels=Y):
    x, y = logits[ix], labels[ix]
    real = x.gather(1, y[:,None]).flatten()
    runner = x.masked_fill(F.one_hot(y, P).bool(), -torch.inf).amax(1)
    return dict(correct=int((x.argmax(1)==y).sum()), ce=float(F.cross_entropy(x,y)),
                margin=float((real-runner).mean()))


def at_head(model, pre):
    return model.classifier(model.final_norm(model.blocks[1].norm(pre)))


def joint(x):
    """Orthogonal interaction in the uniform, fully observed two-operand grid."""
    grid = x.reshape(P,P,*x.shape[1:])
    return (grid-grid.mean(0,keepdim=True)-grid.mean(1,keepdim=True)+grid.mean((0,1),keepdim=True)).reshape_as(x)


@torch.no_grad()
def capture(model):
    local = model.blocks[0]
    rows = {}
    for start in range(0,N,192):
        t = DATA["tokens"][start:start+192]
        r = DATA["roles"][start:start+192]
        h = model.residue_features(t) + model.role_features(r) + model.position[None]
        source = local.drive(h).reshape(len(t),16,M,8)
        right, left = local.sweep(source)
        # Compute the native first layer and retain only the site-7 mode factorization.
        hidden = local(h)
        z, second = bridge.native_parts(model, hidden, "TGLT")
        v = torch.einsum("nd,mdo->nmo",h[:,7],local.visible_factor)
        f = torch.einsum("nmc,mco->nmo",torch.cat((right[:,7],left[:,7]),-1),local.mode_factor)
        mixed = local.local_coupling * local.mode_logits.softmax(0)[None,:,None] * v*f
        w = F.linear(mixed,local.out.weight)
        pre7 = h[:,7]+local.out.bias+w.sum(1)
        assert (local.norm(pre7)-hidden[:,7]).abs().max() < 2e-10
        second_drive = model.blocks[1].drive(hidden).reshape(len(t),16,M,8)
        arrived2 = torch.einsum("mj,njmc->nmc",model.sweep.K_right[:,15],second_drive)
        fields = dict(h=hidden, received2=arrived2, source=source[:,0], arrived=right[:,7], local=h[:,7],
                      mode_w=w, products=mixed, pre7=pre7, pre2=second["pre1"], q=second["final_input"], z=z)
        for name,val in fields.items():
            rows.setdefault(name,[]).append(val.contiguous())
    rows = {key:torch.cat(vals) for key,vals in rows.items()}
    rows["w"] = rows["mode_w"].sum(1)
    rows["j"] = joint(rows["w"])
    lams = model.sweep.lambdas**7
    err = rows["arrived"][DONOR]-rows["arrived"]-(rows["source"][DONOR]-rows["source"])*lams[None,:,None]
    rows["transport_error"] = float(err.abs().max())
    assert rows["transport_error"] < 2e-11
    return rows


def influence_matrix(model, state, pos):
    """Linear map from one changed h1 position to layer-2 pre-norm query, at fixed b."""
    assert pos < 15 and model.sweep.H < 15
    z = state["h"][:P,15]
    assert (state["h"][:,15]-z[B]).abs().max() < 2e-12
    bl = model.blocks[1]
    coeff = torch.einsum("mck,mco,m->mko",bl.drive.weight.reshape(M,8,D),
                         bl.mode_factor[:,:8],bl.sweep.K_right[:,15,pos])
    visible = torch.einsum("bd,mdo->bmo",z,bl.visible_factor) * bl.mode_logits.softmax(0)[None,:,None]
    return bl.local_coupling*torch.einsum("mko,bmo,do->bkd",coeff,visible,bl.out.weight)


def change_pre(state, mat, delta):
    term = torch.einsum("abk,bkd->abd",delta.reshape(P,P,D),mat)
    return state["pre2"]+term.reshape(N,D)


def hidden_output(model, state, mat, pos, new_h):
    return at_head(model,change_pre(state,mat,new_h-state["h"][:,pos]))


def write_output(model,state,mat,delta):
    return hidden_output(model,state,mat,7,model.blocks[0].norm(state["pre7"]+delta))


def ridge_probe(features, label=Y):
    # Relative ridge, affine intercept, unrestricted 59-class targets; training rows only.
    x = features.reshape(N,-1)
    x = x-x[TR].mean(0)
    gram = x[TR].T@x[TR]
    ridge = .001*gram.diag().mean().clamp_min(1e-30)
    gram.diagonal().add_(ridge)
    one = F.one_hot(label[TR],P).double()
    prior = one.mean(0)
    weights = torch.linalg.solve(gram,x[TR].T@(one-prior))
    return int(((x[TE]@weights+prior).argmax(1)==label[TE]).sum())


def operand_count(x):
    x = x.reshape(N,-1)
    one = F.one_hot(A[TR],P).double()
    means = (one.T@x[TR])/one.sum(0)[:,None]
    # Nearest-centroid decisions are a linear classifier with a class-specific intercept.
    logits = 2*x[TE]@means.T - means.square().sum(1)
    return int((logits.argmax(1)==A[TE]).sum())


def kernel_probe(state):
    # Fixed product kernel on two signals available before learned local multiplication.
    kg = []
    for x in (state["arrived"].reshape(N,-1),state["local"]):
        v = x-x[TR].mean(0)
        kg.append((v@v[TR].T)/v[TR].square().sum(1).mean().clamp_min(1e-30))
    k = (1+kg[0])*(1+kg[1])
    gram = k[TR].clone()
    gram.diagonal().add_(.001*gram.diag().mean())
    weights = torch.cholesky_solve(F.one_hot(Y[TR],P).double(),torch.linalg.cholesky(gram))
    return int(((k[TE]@weights).argmax(1)==Y[TE]).sum())


def local_summary(model,s):
    mat = influence_matrix(model,s,7)
    removed_joint = score(write_output(model,s,mat,-s["j"]))
    h7 = s["h"][:,7]
    h7_var = h7-h7.reshape(P,P,D).mean(0)[B]
    pre_no7 = change_pre(s,mat,-h7_var)
    pre_only7 = s["pre2"].reshape(P,P,D).mean(0)[B]+s["pre2"]-pre_no7
    no7 = score(at_head(model,pre_no7))
    only7 = score(at_head(model,pre_only7))
    bl = model.blocks[1]
    visible = torch.einsum("bd,mdo->bmo",s["h"][:P,15],bl.visible_factor)
    vis_centered = (visible-visible.mean(0))[B]
    incoming = s["received2"]-s["received2"].reshape(P,P,M,8).mean(0)[B]
    transformed = torch.einsum("nmc,mco->nmo",incoming,bl.mode_factor[:,:8])
    query_mod = F.linear((bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]*vis_centered*transformed).sum(1),bl.out.weight)
    no_mod = score(at_head(model,s["pre2"]-query_mod))
    native = score(s["z"])
    train = score(s["z"],TR)
    return dict(train_correct=train["correct"], train_ce=train["ce"], train_margin=train["margin"],
        test_correct=native["correct"], test_ce=native["ce"], test_margin=native["margin"],
        source_id=operand_count(s["source"]), received_id=operand_count(s["arrived"]),
        preinteraction_rule=kernel_probe(s), local_products_rule=ridge_probe(s["products"]),
        local_write_rule=ridge_probe(s["w"]), local_state_rule=ridge_probe(s["h"][:,7]),
        query_input_rule=ridge_probe(s["received2"]), query_pre_rule=ridge_probe(s["pre2"]), query_rule=ridge_probe(s["q"]),
        no7_correct=no7["correct"], no7_ce=no7["ce"], no7_margin=no7["margin"],
        only7_correct=only7["correct"], only7_ce=only7["ce"], only7_margin=only7["margin"],
        no_query_mod_correct=no_mod["correct"], no_query_mod_ce=no_mod["ce"], no_query_mod_margin=no_mod["margin"], joint_removed_correct=removed_joint["correct"],
        joint_removed_ce=removed_joint["ce"], joint_removed_margin=removed_joint["margin"],
        transport_energy=float((s["arrived"][DONOR]-s["arrived"]).square().sum((1,2)).mean()),
        interaction_energy=float(s["j"].square().sum(1).mean()), transport_error=s["transport_error"])
