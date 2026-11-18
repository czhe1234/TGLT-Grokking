"""Check the raw histories, original checkpoints, intervention controls, and finite response probes."""
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
N_TRAIN, N_TEST = 870, 2611


def read_csv(name):
    with (OUT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def table(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sustained(points, valid):
    steps = []
    for step, row in points:
        if valid(row):
            steps.append(step)
            if len(steps) == 3:
                return steps[0]
        else:
            steps.clear()
    return None


def main():
    history = read_csv("original_embedding.csv")
    response = read_csv("training_response.csv")
    assert len(history) == 1202 and len(read_csv("original_query.csv")) == 254
    assert len(response) == 300 and len(read_csv("pulse_swaps.csv")) == 60
    historical = {}
    natural = []
    for run in sorted({r["run_id"] for r in history}):
        points = sorted((int(r["step"]), r) for r in history if r["run_id"] == run)
        assert [t for t, _ in points] == list(range(0, 300001, 500))
        historical[run] = dict(points)
        for threshold in (.1, .5, .9):
            threshold_n = math.ceil(N_TEST * threshold)
            natural.append(dict(run=run, threshold=threshold,
                                onset=sustained(points, lambda r: int(r["test_correct"]) >= threshold_n)))
    table("natural_onsets.csv", natural)
    residual_checks = 0
    for file in sorted(OUT.glob("error_swap_*.json")):
        rows = json.loads(file.read_text())
        assert len(rows) == 30
        pairs = {(r["recipient_step"], r["donor_step"]) for r in rows}
        assert len(pairs) == 6
        for t, donor in pairs:
            runs = {r["arm"]:r for r in rows if (r["recipient_step"],r["donor_step"]) == (t,donor)}
            assert set(runs) == {"native","own_residual","zero_current","donor","donor_norm_matched"}
            assert all(runs["native"][k] == runs["own_residual"][k] for k in runs["native"] if k != "arm")
            residual_checks += 1
    assert residual_checks == 12
    analytical_error = [abs(float(r["finite_transfer"])-float(r["transfer"])) for r in response if r["finite_transfer"]]
    assert len(analytical_error) == 24 and max(analytical_error) < 3e-5
    curves, outcomes, matched = [], [], []
    maximum = {"implementation_norm_error": 0., "implementation_radial_error": 0., "implementation_mean_error": 0.}
    checked_versions = set()
    native_checks = []
    source_hashes = {}
    native_weight_errors = []
    for fork in PROTOCOL["forks"]:
        name, run, origin = fork["name"], fork["run"], fork["step"]
        baseline = None
        for arm in PROTOCOL["arms"]:
            track = name + "__" + arm
            dest = HERE / "records" / track
            meta = json.loads((dest/"meta.json").read_text())
            rows = json.loads((dest/"curve.json").read_text())
            complete = json.loads((dest/"complete.json").read_text())
            assert all(complete[k] == v for k, v in meta.items())
            assert complete["steps"] == 4000 and complete["points"] == 41
            assert meta["arm"] == arm and meta["run_id"] == run and meta["fork_step"] == origin
            if meta["code_commit"] not in checked_versions:
                subprocess.run(["git","diff","--exit-code","--quiet",
                                "834f4f76adb1e128c60184d25c78aed49b6b0580", meta["code_commit"],
                                "--", "branch.py","order.py","protocol.json"],cwd=HERE,check=True)
                checked_versions.add(meta["code_commit"])
            assert meta["protocol_sha256"] == digest(HERE/"protocol.json")
            assert meta["source_sha256"] == digest(meta["source"]) == historical[run][origin]["source_sha256"]
            assert meta["uuid"].split("GPU-")[-1] == fork["gpu"][4:]
            source_hashes[name] = meta["source_sha256"]
            steps = [r["relative_step"] for r in rows]
            assert steps == list(range(0, 4001, 100))
            if arm == "native":
                baseline = rows[0]
            assert rows[0] == baseline
            saved = torch.load(meta["source"], map_location="cpu", weights_only=False)
            for rel, fname in [(1000,"pulse_end.pt"), (4000,"latest.pt")]:
                resume = torch.load(HERE/".branch_state"/track/fname, map_location="cpu", weights_only=False)
                assert resume["step"] == origin + rel
                assert resume["config"] == saved["config"] and resume["meta"] == meta
                assert resume["branch_history"] == rows[:rel//100+1]
                assert all(int(s["step"]) == resume["step"] for s in resume["optimizer_state"]["state"].values())
                assert torch.equal(resume["loader_state"], saved["loader_state"])
                assert torch.equal(resume["rng_state"]["cuda"][0], saved["rng_state"]["cuda"][0])
                assert torch.equal(resume["rng_state"]["torch"], saved["rng_state"]["torch"])
                assert resume["rng_state"]["python"] == saved["rng_state"]["python"]
                if arm == "native":
                    p = Path(meta["source"]).with_name("checkpoint_%07d.pt" % (origin+rel))
                    if not p.is_file():
                        with (HERE.parent/"relay_learning_20261118/results/trajectories.csv").open() as f:
                            p = Path(next(q["path"] for q in csv.DictReader(f)
                                          if q["run_id"]==run and int(q["step"]) == origin+rel))
                    assert digest(p) == historical[run][origin+rel]["source_sha256"]
                    actual = torch.load(p, map_location="cpu", weights_only=False)
                    errors = [float((v.double()-actual["model_state"][k].double()).abs().max()) for k,v in resume["model_state"].items()]
                    native_weight_errors.append(dict(fork=name, relative_step=rel, max_abs_weight_difference=max(errors)))
            for r in rows:
                assert r["step"] == origin + r["relative_step"] and all(math.isfinite(v) for v in r.values())
                if r["relative_step"] > 1000 or arm == "native":
                    assert all(r[k] == 0 for k in ["pulse_order_extra_linear","pulse_native_order_linear",*maximum])
                for k in maximum:
                    maximum[k] = max(maximum[k], r[k])
                for k in ("embedding_order","shuffled_order","received_order","local_b_order","query_sum_fraction"):
                    assert 0 <= r[k] <= 1
                if arm == "native" and r["relative_step"]%500 == 0:
                    ref = historical[run][r["step"]]
                    matches = all(r[k] == r["historical_"+k] == int(ref[k]) for k in ("train_correct","test_correct"))
                    native_checks.append(dict(fork=name,relative_step=r["relative_step"],matches=matches,
                                              test_delta=r["test_correct"]-int(ref["test_correct"]),
                                              train_delta=r["train_correct"]-int(ref["train_correct"])))
                curves.append(dict(fork=name,arm=arm,**r))
            observed = [(r["relative_step"],r) for r in rows if r["relative_step"]%500 == 0]
            reached = lambda r: 10*r["test_correct"] >= N_TEST
            query_use = lambda r: 20*(r["test_correct"]-r["query_removed_correct"]) >= N_TEST
            event = sustained(observed, lambda r: reached(r) and query_use(r))
            release = rows[10]
            later = [r for r in rows if r["relative_step"] > 1000]
            total_pulse = sum(r["pulse_order_extra_linear"] for r in rows)
            if arm == "increase":
                assert total_pulse > 0
            if arm == "decrease":
                assert total_pulse < 0
            outcomes.append(dict(fork=name,run_id=run,fork_step=origin,arm=arm,
                                 primary_onset=event,primary_confirmation=None if event is None else event+1000,
                                 window_end=4000,native_accuracy_onset=sustained(observed, reached),
                                 release_train=release["train_correct"],release_test=release["test_correct"],
                                 release_embedding_order=release["embedding_order"],release_same_sum=release["query_sum_fraction"],
                                 release_test_ce=release["test_ce"],release_test_margin=release["test_margin"],
                                 mean_after_release_test_ce=statistics.mean(r["test_ce"] for r in later),
                                 mean_after_release_test_margin=statistics.mean(r["test_margin"] for r in later),
                                 min_train_correct=min(r["train_correct"] for r in rows),
                                 mean_after_release_test_pct=100*statistics.mean(r["test_correct"] for r in later)/N_TEST,
                                 mean_after_release_no_query_pct=100*statistics.mean(r["query_removed_correct"] for r in later)/N_TEST,
                                 pulse_linear_order_difference=total_pulse))
        probe = next(r for r in response if r["scope"] == "all" and r["force"] == "true" and int(r["step"]) == origin and r["run_id"] == run)
        matched.append(dict(fork=name,step=origin,train_correct=baseline["train_correct"],test_correct=baseline["test_correct"],
                            embedding_order=baseline["embedding_order"],same_sum=baseline["query_sum_fraction"],
                            transfer=float(probe["transfer"]),cosine=float(probe["cosine"]),
                            with_query_removed=float(probe["finite_no_query_transfer"])))
    assert len(native_checks)==27 and len(curves)==492 and max(maximum.values())<.0002
    table("branch_curves.csv",curves)
    table("branch_outcomes.csv",outcomes)
    table("matched_states.csv",matched)
    native_errors = [v for v in native_checks if not v["matches"]]
    v = dict(completed_arms=len(outcomes),observations=len(curves),original_snapshots=len(history),
             native_historical_points=len(native_checks),native_historical_disagreements=native_errors,
             native_restored_parameter_checks=native_weight_errors,source_sha256=source_hashes,rounding_error_max=maximum,
             original_adam_error_replays_checked=residual_checks, original_adam_error_replay_arms=5*residual_checks,
             post_release_steps=3000,minimum_train_correct=min(r["train_correct"] for r in curves),
             response_finite_checks=len(analytical_error),response_max_error=max(analytical_error),
             swanlab=json.loads((OUT/"swanlab_validation.json").read_text()))
    (OUT/"validation.json").write_text(json.dumps(v,indent=2)+"\n")
    print(json.dumps(v,indent=2))


if __name__ == "__main__":
    main()
