"""Validate intervention results and extract sustained functional differences from all saved states."""
import csv
import json
from collections import Counter
from pathlib import Path
from replay import *


def read_rows(path):
    return list(csv.DictReader(path.open()))


def main():
    rows=read_rows(OUT/"trajectories.csv")
    assert len(rows)==1404
    trajectory={}
    for row in rows:
        key=row["run_id"],int(row["step"])
        assert key not in trajectory
        trajectory[key]=row
        assert digest(Path(row["path"]))==row["sha256"]
        assert int(row["received_id"])==int(row["source_id"])==2611
    ups=read_json(OUT/"updates.json")
    native=read_json(OUT/"native_effects.json")
    controls=read_json(OUT/"controls.json")
    patches=read_json(OUT/"relay.json")
    hybrids=read_rows(OUT/"cross_time.csv")
    keys={(rid,t) for i,rid in enumerate(RUNS) for t in ANCHORS[i//2]}
    for data in (ups,native,controls,patches):
        assert len(data)==16 and {(r["run_id"],r["step"]) for r in data}==keys
    assert len(hybrids)==64
    diffs=[]
    for width,(fast,slow) in zip((12,8),zip(RUNS[::2],RUNS[1::2])):
        steps=sorted(t for r,t in trajectory if r==fast)
        assert len(steps)==(601 if width==12 else 101)
        for metric in ("no_query_mod_correct","joint_removed_correct","no7_correct"):
            for threshold in (3,5,10,20):
                def cost(rid,t):
                    r=trajectory[rid,t]
                    return 100*(int(r["test_correct"])-int(r[metric]))/2611
                def passes(t):
                    c=cost(fast,t)
                    return c>=threshold and c-cost(slow,t)>=threshold
                first=next((t for i,t in enumerate(steps[:-2]) if all(passes(x) for x in steps[i:i+3])),None)
                diffs.append(dict(H=width,operation=metric,min_loss_and_gap_pp=threshold,consecutive=3,first_saved_step=first,
                    fast_loss_pp=cost(fast,first) if first is not None else None,
                    slow_loss_pp=cost(slow,first) if first is not None else None))
    table(OUT/"functional_divergence.csv",diffs)
    fp=[r for chip in ("A800","H100") for r in read_json(OUT/("precision_"+chip+".json"))]
    changed={(r["run_id"],int(r["step"])) for r in rows if int(r["replay_delta_train"]) or int(r["replay_delta_test"])}
    assert {(r["run_id"],r["step"]) for r in fp}==changed
    assert all(r["before"]["test"]["correct"]==r["original_test_correct"] for r in fp)
    first_steps=[r for chip in ("A800","H100") for r in read_json(OUT/("native_"+chip+".json")) if r["step"]==0]
    assert len(first_steps)==4 and all(r["first_historical_step_verified"] for r in first_steps)
    for u,n,c,b in zip(sorted(ups,key=lambda x:(x["run_id"],x["step"])), sorted(native,key=lambda x:(x["run_id"],x["step"])),
                         sorted(controls,key=lambda x:(x["run_id"],x["step"])),sorted(patches,key=lambda x:(x["run_id"],x["step"]))):
        rid,t=u["run_id"],u["step"]
        a=trajectory[rid,t]
        assert int(a["test_correct"])==int(a["original_test_correct"])==u["base"]["test_correct"]==n["native_before"]["test"]["correct"]
        assert int(a["train_correct"])==int(a["original_train_correct"])==u["base"]["train_correct"]==n["native_before"]["train"]["correct"]
        assert len(u["delete_before"])==len(n["deletion_after"])==16
        assert all(int(a[key])==u["base"][key] for key in ("query_rule","joint_removed_correct","no_query_mod_correct"))
        query=next(x for x in c["counterfactuals"] if x["condition"]=="remove_query_mod_15")
        assert query["test_correct"]==u["base"]["no_query_mod_correct"]
        assert len([x for x in b["patches"] if x["condition"]=="single_site" and x["split"]=="held"])==16
        assert all(x["n"]==(2611 if x["split"]=="held" else 1941) for x in b["patches"])
        assert n["parameter_decomposition_exact"]
        assert u["update"]["parameter_replay_error"]<1e-12
    for rid in RUNS:
        rx=[h for h in hybrids if h["run_id"]==rid]
        for idx,tag in ((0,"early_step"),(15,"late_step")):
            h=next(z for z in rx if int(z["mask"])==idx)
            assert int(h["test_correct"])==int(trajectory[rid,int(h[tag])]["test_correct"])
    ep=next(r for r in patches if r["run_id"]==RUNS[2] and r["step"]==100000)
    byname={x["condition"]:x for x in ep["repairs"]}
    assert byname["untouched"]["test_correct"]==byname["exact_restore_check"]["test_correct"]==2607
    assert byname["delete_u9"]["test_correct"]==1526
    assert byname["surviving_write"]["agrees_native"]==2611
    max_err=max(r["checks"]["sum_sites_error"] for r in patches)
    assert max_err<1e-7
    report=dict(runs=4,saved_states=len(rows),anchor_updates=len(keys),historical_step1_checks=4,
        named_checkpoints_match_history=True,train_test_n=[870,2611],
        cpu_counts_matched=len(rows)-len(changed),cpu_mismatches_checked_on_original_gpu=len(fp),
        cpu_max_test_count_difference=max(abs(int(r["replay_delta_test"])) for r in rows),
        original_gpu_count_mismatches=0,max_delta_transport_identity=max(float(r["transport_error"]) for r in rows),
        position_patch_decomposition_max_abs_error=max_err,altered_forward_reference_max_abs_error=max(r["update"]["native_edit_error"] for r in ups),
        max_fp64_adamw_formula_error=max(r["update"]["parameter_replay_error"] for r in ups),
        original_precision_group_decomposition="exact at all 16 checkpoints",checkpoint_sha256_verified=len(rows),
        hybrids_within_seed=64,independent_run_pairs=2)
    write_json(OUT/"validation.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
