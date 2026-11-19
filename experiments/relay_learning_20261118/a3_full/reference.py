"""Recreate the previous equivalent-path logits at exactly the old weights (CPU, FP64)."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import replay as old
import torch
import torch.nn.functional as F
from controls import matched

CONTROL_NAMES = ["no7_a", "no7_joint", "no_query", "wrong_query_b1"]


@torch.inference_mode()
def references(model):
    s = old.capture(model)
    mat = old.influence_matrix(model, s, 7)
    ws = s["mode_w"]
    results = []
    for mask in range(16):
        ids = [j for j in range(4) if mask & (1 << j)]
        if ids:
            z = old.write_output(model, s, mat, -ws[:, ids].sum(1))
        else:
            z = s["z"]
        results.append(z)
    h7 = s["h"][:, 7]
    mean_h7 = h7.reshape(59, 59, -1).mean(0)[old.B]
    results.append(old.hidden_output(model, s, mat, 7, mean_h7))
    results.append(old.write_output(model, s, mat, -s["j"]))
    bl = model.blocks[1]
    vis = torch.einsum("bd,mdo->bmo", s["h"][:59, 15], bl.visible_factor)
    vis = vis - vis.mean(0)
    signal = s["received2"]
    signal = signal - signal.reshape(59, 59, 4, 8).mean(0)[old.B]
    term = torch.einsum("nmc,mco->nmo", signal, bl.mode_factor[:, :8])
    term = term * bl.local_coupling * bl.mode_logits.softmax(0)[None, :, None]
    correct = F.linear(term * vis[old.B], bl.out.weight).sum(1)
    wrong = F.linear(term * vis[(old.B+1) % 59], bl.out.weight).sum(1)
    results.append(old.at_head(model, s["pre2"] - correct))
    results.append(old.at_head(model, s["pre2"] - correct + matched(wrong, correct)))
    return torch.stack(results).numpy()


def make(job):
    rid, step = job
    torch.set_num_threads(1)
    model, _, _, _, _ = old.load(rid, step)
    for stage in ("before", "after"):
        target = HERE / ".work" / f"ref_{rid}_{step}_{stage}.npy"
        if stage == "after":
            cache = old.HERE / ".native_cache" / f"{rid}_{step:07d}.pt"
            after = torch.load(cache, map_location="cpu", weights_only=False)["after"]
            params = {n:p for n, p in model.named_parameters() if p.requires_grad}
            assert params.keys() == after.keys()
            with torch.no_grad():
                for n,p in params.items():
                    p.copy_(after[n])
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.save(f, references(model), allow_pickle=False)
            os.replace(tmp, target)
    return rid, step


if __name__ == "__main__":
    (HERE / ".work").mkdir(exist_ok=True)
    jobs = [(rid, step) for i, rid in enumerate(old.RUNS) for step in old.ANCHORS[i//2]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for rid, step in pool.map(make, jobs):
            print(rid, step, flush=True)
