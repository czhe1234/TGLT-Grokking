"""Compare the specified states; export A3 and the existing experiment B measurements."""
import csv
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
OLD=HERE.parent/"results"

def load_csv(path):
    with path.open(encoding="utf-8") as f: return list(csv.DictReader(f))

def write_csv(name,rows):
    path=OUT/name
    with path.with_suffix(".tmp").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    path.with_suffix(".tmp").replace(path)

def main():
    expected={r["run_id"]:{4000,107000,155000,300000} if r["run_id"].startswith("b") else {3000,51000,58000,100000} for r in json.loads((OLD/"native_effects.json").read_text())}
    keys={(rid,s) for rid,steps in expected.items() for s in steps}
    states=[json.loads((HERE/".work"/f"{rid}_{s}.json").read_text()) for rid,s in sorted(keys)]
    assert len(states)==16 and all((r["run_id"],r["step"]) in keys for r in states)
    modes=[];shifts=[];tests=[];sources=[];sums=[];rank_changes=[]
    for state in states:
        rid,step=state["run_id"],state["step"]
        tag=dict(run_id=rid,step=step)
        sources.append({k:v for k,v in state.items() if k not in ("modes","offsets","deletions")})
        for stage in ("before","after"):
            m=[dict(**tag,**r) for r in state["modes"] if r["stage"]==stage]
            s=[dict(**tag,**r) for r in state["offsets"] if r["stage"]==stage]
            d=[dict(**tag,**r) for r in state["deletions"] if r["stage"]==stage]
            assert {r["mode"] for r in m}=={1,2,3,4} and len(m)==4
            assert {(r["mode"],r["offset"]) for r in s}=={(j,k) for j in range(1,5) for k in range(1,59)} and len(s)==232
            assert len(d)==20 and len({r["condition"] for r in d})==20
            assert all(r["pairs"]==59**2*58 and r["indistinguishable_pairs"]==0 for r in m)
            assert max(abs(r["previous_transport_adj"]-r["transport_adj"])/r["transport_adj"] for r in m)<1e-7
            assert max(abs(r["previous_mixed_adj"]-r["mixed_adj_adj"])/max(1e-25,r["mixed_adj_adj"]) for r in m)<1e-6
            assert all(r["read_correct"]==2611 and r["combined_read_correct"]==2611 for r in m)
            baseline=next(r for r in d if r["condition"]=="mask_00")
            for r in d:
                for pref in ("prev","direct64","direct32"):
                    r[pref+"_acc_loss_pp"]=baseline[pref+"_test_pct"]-r[pref+"_test_pct"]
                    r[pref+"_ce_cost"]=r[pref+"_test_ce"]-baseline[pref+"_test_ce"]
                    r[pref+"_margin_loss"]=baseline[pref+"_test_margin"]-r[pref+"_test_margin"]
            order=lambda k:[r["mode"] for r in sorted(m,key=lambda v:v[k],reverse=True)]
            if order("transport_adj")!=order("transport_all"): rank_changes.append([rid,step,stage])
            sums.append(dict(**tag,stage=stage,transport_adj=sum(r["transport_adj"] for r in m),
                             transport_all=sum(r["transport_all"] for r in m),mixed_adj_adj=sum(r["mixed_adj_adj"] for r in m),
                             mixed_all_a_adj_b=sum(r["mixed_all_a_adj_b"] for r in m),mixed_all_a_all_b=sum(r["mixed_all_a_all_b"] for r in m),
                             combined_read_pct=100* m[0]["combined_read_correct"]/2611))
            modes.extend(m); shifts.extend(s); tests.extend(d)
    for name,rows in [("sources.csv",sources),("modes.csv",modes),("replacement_offsets.csv",shifts),
                      ("full_forward.csv",tests),("state_summary.csv",sums)]:
        write_csv(name,rows)
    stats={}
    for group,rr in (("mode_masks",[r for r in tests if r["condition"].startswith("mask")]),
                     ("mechanism_controls",[r for r in tests if not r["condition"].startswith("mask")])):
        stats[group]={}
        for pref in ("direct64","direct32"):
            stats[group][pref]=dict(conditions=len(rr),full_input_rows=len(rr)*3481,
                different_predictions=sum(r[pref+"_all_predictions_changed"] for r in rr),
                different_held_predictions=sum(r[pref+"_held_predictions_changed"] for r in rr),
                max_correct_count_difference=max(abs(r[pref+"_test_correct"]-r["prev_test_correct"]) for r in rr),
                max_logit_error=max(r[pref+"_max_logit_error"] for r in rr),
                max_held_ce_difference=max(abs(r[pref+"_test_ce"]-r["prev_test_ce"]) for r in rr))
        assert stats[group]["direct64"]["different_predictions"]==0
    report=dict(anchors=16,weight_states=32,ordered_pairs_per_state=3481*58,pairs_per_fixed_b=59*58,
                total_ordered_pair_observations=32*3481*58,total_masks=512,
                ratio_range=[min(r["all_over_adj"] for r in modes),max(r["all_over_adj"] for r in modes)],
                ratio_median=median(r["all_over_adj"] for r in modes),transport_rank_changes=rank_changes,
                transport_identity_max_error=max(r["identity_max_abs"] for r in modes),
                source_hash="e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73",comparison=stats)
    (OUT/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    result_b=[]
    for r in load_csv(OLD/"patches.csv"):
        if r["run_id"] not in ("r2_T4_M4_seed0","r2_T4_M4_seed1") or r["step"]!="100000": continue
        if r["split"]=="both_held" and r["condition"]=="single_site" and r["position"]=="7":
            n=int(r["n"])
            result_b.append(dict(run_id=r["run_id"],condition="replace_site7",n=n,
                        correspond_to="donor",correct=int(r["donor_true"]),correct_pct=100*int(r["donor_true"])/n,
                        agree=int(r["donor_prediction"]),agree_pct=100*int(r["donor_prediction"])/n))
    for r in load_csv(OLD/"repairs.csv"):
        if r["run_id"]=="r2_T4_M4_seed1" and r["step"]=="100000":
            n=int(r["test_n"])
            result_b.append(dict(run_id=r["run_id"],condition=r["condition"],n=n,correspond_to="original",
                        correct=int(r["test_correct"]),correct_pct=100*int(r["test_correct"])/n,
                        agree=int(r["agrees_native"]),agree_pct=100*int(r["agrees_native"])/n))
    write_csv("experiment_B_percentages.csv",result_b)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
