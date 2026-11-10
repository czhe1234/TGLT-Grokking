from pathlib import Path
import ast
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path("/home/chenzhe/work/tglt_experiments")
OUT = Path(__file__).resolve().parent
BASE = ROOT / "week/v1"
AUDIT = ROOT / "analysis/firstgen_addition_numbered_label_audit_20261026/all_192_numbered.csv"
CLASSIFIER = ROOT / "analysis/firstgen_beta_wd_phase_20261027/source/strict_label_source.py"
R_VALUES = (1, 2, 3, 4)
T_VALUES = (3, 4, 5, 7)
M_VALUES = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
STEPS = np.array([0, 1, *range(500, 100001, 500)], dtype=int)
BLUE = "#115D9C"
RED = "#C74124"
INK = "#192733"
SUBTLE = "#475663"
FONT = "Noto Sans CJK JP"
plt.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#65727b", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": SUBTLE, "ytick.color": SUBTLE,
    "savefig.facecolor": "white", "font.size": 10, "figure.dpi": 130,
})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def csv_out(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_audit():
    src = CLASSIFIER.read_text()
    function = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "audit")
    scope = {"math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CLASSIFIER), "exec"), scope)
    return scope["audit"]


def interpolated_center(time, test):
    ix = np.flatnonzero(test >= 0.5)
    if not len(ix):
        return None
    k = int(ix[0])
    if k == 0:
        return float(time[k])
    assert test[k-1] < 0.5 <= test[k]
    center = time[k-1] + (0.5-test[k-1]) * (time[k]-time[k-1]) / (test[k]-test[k-1])
    assert abs(np.interp(center, time, test)-0.5) < 1e-11
    return float(center)


def mean_curves(records, align):
    series = [(a["time"]-a["center"] if align else a["time"], a["train"], a["test"]) for a in records]
    if align:
        lo = max(float(v[0][0]) for v in series)
        hi = min(float(v[0][-1]) for v in series)
        assert lo <= 0 <= hi
        points = np.unique(np.concatenate([[lo, 0.0, hi], *(a[0] for a in series)]))
        points = points[(points >= lo) & (points <= hi)]
    else:
        points = STEPS
    def reduce(k):
        values = np.stack([np.interp(points, run[0], run[k]) for run in series]) if align else np.stack([run[k] for run in series])
        assert values.shape == (3, len(points))
        return values.mean(axis=0), values.std(axis=0, ddof=1)
    tr, trs = reduce(1)
    te, tes = reduce(2)
    if align:
        assert abs(te[points == 0].item()-0.5) < 1e-11 and tes[points == 0].item() < 1e-11
    return (points, tr, trs, te, tes)


def gather():
    with AUDIT.open(encoding="utf-8-sig") as f:
        roster = {x["run_id"]: x for x in csv.DictReader(f)}
    manifest = json.loads((BASE / "run_manifest.json").read_bytes())
    expected = set(itertools.product(R_VALUES, T_VALUES, M_VALUES, SEEDS))
    assert len(roster) == 192 and len(manifest["runs"]) == 192
    assert {(q["r"],q["T"],q["M"],q["seed"]) for q in manifest["runs"]} == expected
    audit = read_audit()
    runs, groups = [], {}
    for r,t,m,seed in sorted(expected):
        run_id = f"r{r}_T{t}_M{m}_seed{seed}"
        old = roster[run_id]
        base = BASE / "results" / run_id
        cbytes = (base/"config.json").read_bytes()
        hbytes = (base/"history.json").read_bytes()
        cfg = json.loads(cbytes)
        history = json.loads(hbytes)
        assert digest(cbytes) == old["config_sha256"] and digest(hbytes) == old["history_sha256"], run_id
        expected_cfg = dict(r=r,T=t,M=m,seed=seed,task="add",p=59,train_fraction=.25,
                            local_coupling=1.,wd=.1,layers=2,pos_a=0,pos_b=7,query_pos=15,
                            seq_len=16,steps=100000,block_type="full",random_labels=False)
        assert all(cfg[k] == v for k,v in expected_cfg.items()), run_id
        time = np.array([y["step"] for y in history]); assert np.array_equal(time, STEPS), run_id
        for y in history:
            assert y["train_n"] == 870 and y["test_n"] == 2611, run_id
            for split in ("train","test"):
                assert abs(y[f"{split}_acc"]-y[f"{split}_correct"]/y[f"{split}_n"]) < 1e-14, run_id
        train = np.array([y["train_acc"] for y in history]); test = np.array([y["test_acc"] for y in history])
        cls = audit(history)
        assert cls["audit_fixed_grokking"] == (old["old_fixed_grokking"] == "True") == (old["audit_fixed_grokking"] == "True"), run_id
        for k in ("t_mem","t_low_end","t90","plateau_length","jump_width"):
            ref = None if old[k]=="" else int(old[k])
            assert cls[k] == ref, (run_id,k,cls[k],ref)
        center = interpolated_center(time, test)
        record = dict(run_id=run_id,r=r,T=t,M=m,seed=seed,H=r*t,test_ever90=int(test.max()>=.9),
                      strict_grokking=int(cls["audit_fixed_grokking"]),test100k_ge90=int(test[-1]>=.9),
                      final_train_acc=float(train[-1]),final_test_acc=float(test[-1]),peak_test_acc=float(test.max()),
                      t_mem=cls["t_mem"],t50_observed=cls["t50"],t50_interpolated=center,
                      t90_observed=cls["t90"],plateau_steps=cls["plateau_length"],
                      jump_width=cls["jump_width"],strict_reason=cls["reason"],
                      samples=len(time),config_path=str(base/"config.json"),history_path=str(base/"history.json"),
                      config_sha256=digest(cbytes),history_sha256=digest(hbytes))
        runs.append(record)
        groups.setdefault((r,t,m),[]).append(dict(time=time,train=train,test=test,center=center,run=record))
    assert all(len(v)==3 and [z["run"]["seed"] for z in v]==list(SEEDS) for v in groups.values())
    return runs, groups


def group_counts(groups):
    cells, key_map = [], {}
    for (r,t,m), group in groups.items():
        members = [q["run"] for q in group]
        cell = dict(r=r,T=t,M=m,H=r*t,n=3,
                    n_ever90=sum(a["test_ever90"] for a in members),
                    n_strict=sum(a["strict_grokking"] for a in members),
                    n_final90=sum(a["test100k_ge90"] for a in members),
                    n_t50=sum(a["center"] is not None for a in group))
        cell.update({f"seed{a['seed']}_strict":a["strict_grokking"] for a in members})
        cell.update({f"seed{a['seed']}_final_test_acc":a["final_test_acc"] for a in members})
        cells.append(cell)
        key_map[(r,t,m)]=cell
    return cells, key_map


def draw_counts(cells):
    shades = ["#F0F2F4", "#B6D9ED", "#4088B3", "#0A456D"]
    cmap = ListedColormap(shades)
    norm = BoundaryNorm(np.arange(-.5,4.5), cmap.N)
    fig, axes = plt.subplots(2,4, figsize=(12.7,6.55))
    fig.subplots_adjust(left=.105,right=.984,top=.82,bottom=.076,wspace=.23,hspace=.53)
    fig.text(.029,.947,"传播设置与测试结果",fontsize=21,weight="bold")
    fig.text(.030,.893,"每格 3 个 seed。格内数字是满足该行条件的 seed 数。",fontsize=12,color=SUBTLE)
    rownames=("测试达到 90%","严格 Grokking")
    for i,field in enumerate(("n_ever90","n_strict")):
        for j,m in enumerate(M_VALUES):
            ax=axes[i,j]
            z = np.array([[cells[(r,t,m)][field] for t in T_VALUES] for r in R_VALUES])
            ax.imshow(z,cmap=cmap,norm=norm,aspect="auto",interpolation="nearest")
            for rr in range(4):
                for cc in range(4):
                    n=int(z[rr,cc]); ax.text(cc,rr,str(n),ha="center",va="center",weight="bold",fontsize=14,color="white" if n>=2 else INK)
            ax.set_xticks(np.arange(4),[str(x) for x in T_VALUES])
            ax.set_yticks(np.arange(4),[str(x) for x in R_VALUES])
            ax.tick_params(axis="both",length=0,pad=4,labelsize=10)
            ax.set_xticks(np.arange(.5,3.5),minor=True); ax.set_yticks(np.arange(.5,3.5),minor=True)
            ax.grid(which="minor",color="white",linewidth=1.6); ax.tick_params(which="minor",length=0)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_title(f"M = {m}",fontsize=13,weight="bold",pad=5)
            ax.set_xlabel("传播轮数 T",labelpad=1) if i==1 else None
            if j==0: ax.set_ylabel("传播半径 r",labelpad=3)
        fig.text(.028,.663 if i==0 else .237,rownames[i],rotation=90,va="center",ha="center",fontsize=13,weight="bold")
    fig.savefig(OUT/"counts_64.png",dpi=170)
    plt.close(fig)


def draw_means(m,cells,curves,view):
    hybrid=(view=="center")
    fig, axes=plt.subplots(4,4,figsize=(13.2,8.7),sharey=True)
    fig.subplots_adjust(left=.078,right=.984,top=.84,bottom=.093,wspace=.19,hspace=.59)
    tag="首次 50% 对齐" if hybrid else "原始训练步数"
    fig.text(.035,.948,f"{m} 个 mode   ·   三个 seed 的准确率均线",fontsize=21,weight="bold")
    fig.text(.036,.904, f"{tag}。阴影为三个 seed 的标准差。",fontsize=12,color=SUBTLE)
    handles=[Line2D([],[],color=BLUE,lw=2.2,label="训练准确率"),Line2D([],[],color=RED,lw=2.2,label="测试准确率")]
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.987,.985),ncol=2,frameon=False,fontsize=11)
    for i,r in enumerate(R_VALUES):
        for j,t in enumerate(T_VALUES):
            ax=axes[i,j]; cell=cells[(r,t,m)]
            aligned=hybrid and cell["n_t50"]==3
            pts,tr,trs,te,tes=curves[(r,t,m,aligned)]
            x=pts/1000.
            ax.set_axisbelow(True)
            ax.grid(alpha=.67,color="#DADFE2",linewidth=.6)
            if hybrid and not aligned: ax.set_facecolor("#FAF5EB")
            for mean,sd,color in ((tr,trs,BLUE),(te,tes,RED)):
                ax.fill_between(x,np.clip(mean-sd,0,1)*100,np.clip(mean+sd,0,1)*100,color=color,alpha=.13,linewidth=0)
                ax.plot(x,mean*100,color=color,linewidth=1.55,solid_capstyle="round")
            extra = ("  ·  已对齐" if aligned else "  ·  原始") if hybrid else ""
            ax.set_title(f"r={r}   T={t}   H={r*t}{extra}",fontsize=10.8,pad=4)
            ax.set_ylim(-3,104); ax.set_yticks([0,50,100])
            if aligned:
                ax.set_xlim(-100,100); ax.set_xticks([-100,-50,0,50,100])
                ax.axvline(0,color="#73808A",lw=.85,zorder=0)
            else:
                ax.set_xlim(0,100); ax.set_xticks([0,25,50,75,100])
            ax.tick_params(length=2.5,labelsize=8.8,pad=2)
            if j==0: ax.set_ylabel("准确率（%）",fontsize=9)
    caption="每格使用三个 seed 的完整训练记录。横轴为训练步数（千步）。"
    if hybrid:
        caption="白底：按每个 seed 首次达到 50% 对齐，仅画三条共同覆盖的时间段。米色底：有 seed 未到 50%，按原始步数。单位为千步。"
    fig.text(.5,.027,caption,ha="center",fontsize=10.7,color=SUBTLE)
    fig.savefig(OUT/f"means_M{m}_{view}.png",dpi=155)
    plt.close(fig)


def main():
    runs,groups = gather()
    cells,cell_map = group_counts(groups)
    curves={}
    csv_out(OUT/"runs_192.csv",runs)
    csv_out(OUT/"configs_64.csv",cells)
    with gzip.open(OUT/"means_points.csv.gz","wt",encoding="utf-8",newline="") as fp:
        writer=csv.writer(fp)
        writer.writerow(["r","T","M","alignment","n_seeds","step_or_relative_step","train_mean","train_sample_sd","test_mean","test_sample_sd"])
        for (r,t,m),records in groups.items():
            for align in ([False,True] if all(v["center"] is not None for v in records) else [False]):
                key=(r,t,m,align)
                curves[key]=mean_curves(records,align)
                for point in zip(*curves[key]):
                    writer.writerow([r,t,m,"translated_t50" if align else "raw",3,*point])
    draw_counts(cell_map)
    for m in M_VALUES:
        for view in ("raw","center"):
            draw_means(m,cell_map,curves,view)
    def stats(rows):
        return dict(n=len(rows),ever90=sum(a["test_ever90"] for a in rows),
                    strict=sum(a["strict_grokking"] for a in rows),
                    final90=sum(a["test100k_ge90"] for a in rows))
    summary=dict(overall=stats(runs),by_mode={m:stats([a for a in runs if a["M"]==m]) for m in M_VALUES},
                 by_H={h:stats([a for a in runs if a["H"]==h]) for h in sorted({q["H"] for q in runs})},
                 within_ever90_vs_final90_mismatches=sum(a["test_ever90"]!=a["test100k_ge90"] for a in runs),
                 grouped_by_strict_seed_count=dict(sorted(Counter(a["n_strict"] for a in cells).items())),
                 grouped_by_t50_seed_count=dict(sorted(Counter(a["n_t50"] for a in cells).items())),
                 fully_strict_configs=[[a["r"],a["T"],a["M"]] for a in cells if a["n_strict"]==3],
                 archived_label_and_raw_input_hash_matches=192,points_per_run=len(STEPS),history_rows=len(STEPS)*len(runs),
                 native_source=str(BASE/"code/exp2_sequence_local_grokking.py"),
                 native_source_sha256=digest((BASE/"code/exp2_sequence_local_grokking.py").read_bytes()),
                 label_source=str(CLASSIFIER),label_source_sha256=digest(CLASSIFIER.read_bytes()),checked_roster=str(AUDIT),
                 roster_sha256=digest(AUDIT.read_bytes()),manifest=str(BASE/"run_manifest.json"),
                 history_accuracy_reconstruction="correct count divided by split size for every recorded step",
                 labels="train>=.99, consecutive test<=.10 from memorization for >=10000 steps, first test>=.90 with positive transition width strictly less than half the plateau",
                 alignment="At least 50% on all three seeds: linear interpolation for individual t50, translation only, all three present on every averaged point. In the alternate view other cells always retain all three on raw time.",
                 shaded_band="sample standard deviation among three seeds, clipped to accuracy range only for display")
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("overall","grouped_by_t50_seed_count","within_ever90_vs_final90_mismatches","fully_strict_configs")},ensure_ascii=False))


if __name__=="__main__":
    main()
