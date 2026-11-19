"""Read the two historical cohorts at their common saved steps and at initialization, if saved."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path

import replay as rp
import torch

def read_json(path):
    return json.loads(path.read_text())

def original_runs(root):
    july = root / "exp2_no_mlp/current/results/01_classic_grokking/20260712_classic_p59_s16_M4_T4_3seed"
    for seed in range(3):
        yield "july_three_seed", 8, seed, july / f"p59_s16_M4_tr025_wd030_seed{seed}", (10000, 100000)
    for seed in range(3):
        yield "v1_h15", 15, seed, root / "week/v1/results" / f"r3_T5_M4_seed{seed}", (0, 10000, 100000)

@torch.inference_mode()
def measure(path, history, split):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert ck["step"] == history["step"]
    assert (split["pos_a"], split["pos_b"], split["query_pos"]) == (0, 7, 15)
    assert (cfg["seq_len"], cfg["p"], cfg["M"], cfg["layers"], cfg["d_a"]) == (16, 59, 4, 2, 8)
    params = {k: v for k, v in cfg.items() if k in inspect.signature(rp.ref.SequenceLocalModeModel).parameters}
    params["query_pos"] = split["query_pos"]
    params["lambdas"] = cfg["lambda_values"] or rp.ref.xi_spread_lambdas(cfg["M"], cfg["xi_min"], cfg["xi_max"])
    model = rp.ref.SequenceLocalModeModel(**params).double().eval()
    model.load_state_dict(ck["model_state"], strict=True)
    layer = model.blocks[0]
    arrived, predictions = [], []
    for tokens, roles in zip(rp.DATA["tokens"].split(192), rp.DATA["roles"].split(192)):
        h = model.residue_features(tokens) + model.role_features(roles) + model.position[None]
        signal, _ = layer.sweep(layer.drive(h).reshape(len(h), 16, cfg["M"], cfg["d_a"]))
        arrived.append(signal[:, 7])
        predictions.append(model(tokens, roles).argmax(1))
    te = rp.TE
    n = len(te)
    raw_hits = round(history["test_acc"] * n)
    assert abs(raw_hits / n - history["test_acc"]) < 1e-10
    assert history.get("test_correct", raw_hits) == raw_hits
    hits = int((torch.cat(predictions)[te] == rp.Y[te]).sum())
    if hits != raw_hits:
        raise ValueError(f"{path}: replay {hits} differs from original log {raw_hits}")
    return dict(read_a_correct=rp.operand_count(torch.cat(arrived)), sum_correct=raw_hits,
                test_size=n, train_size=len(rp.TR), right_activation_dim=cfg["M"]*cfg["d_a"],
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main(output):
    torch.set_num_threads(1)
    rows = []
    for cohort, horizon, seed, directory, steps in original_runs(rp.ROOT):
        split = read_json(directory / "split.json")
        assert split["train_idx"] == rp.TR.tolist() and split["test_idx"] == rp.TE.tolist(), str(directory)
        cfg = read_json(directory / "config.json")
        assert cfg["r"] * cfg["T"] == horizon and cfg["seed"] == seed
        assert (directory / "checkpoint_0000000.pt").exists() == (0 in steps)
        history = {r["step"]: r for r in read_json(directory / "history.json")}
        for step in steps:
            path = directory / f"checkpoint_{step:07d}.pt"
            row = dict(cohort=cohort, horizon=horizon, seed=seed, step=step,
                       source=str(directory.relative_to(rp.ROOT)))
            row.update(measure(path, history[step], split))
            rows.append(row)
            print(cohort, seed, step, row["read_a_correct"], row["sum_correct"], flush=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
