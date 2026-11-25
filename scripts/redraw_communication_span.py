#!/usr/bin/env python3
"""Redraw the communication-span figure from the histories in this repository."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from strict_grokking import audit

INK = "#263746"
MUTED = "#52616D"
NARROW = "#416581"
WIDE = "#A5553A"
WIDE_DASH = (0, (4, 2))
LIGHT = "#B4BEC5"
BASE = "#8997A2"
ACCENT = "#6E5882"
THIN = (0, (2, 2))

# Archived local pairs with identical initialization, data split, device, and
# numerical settings. The fields are modes, seed, and the wider radius/sweeps.
LOCAL_PAIRS = (
    (1, 0, 4, 4),
    (1, 1, 4, 4),
    (1, 2, 4, 4),
    (2, 0, 4, 4),
    (2, 1, 3, 5),
    (2, 2, 4, 4),
    (4, 1, 3, 5),
    (4, 2, 4, 4),
)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_scan():
    index = {r["run_id"]: r for r in csv_rows(ROOT / "experiments/run_index.csv")
             if r["family"] == "architecture_scan"}
    archived = csv_rows(ROOT / "results/original_architecture_scan_192/runs_192.csv")
    groups = [[], [], []]
    histories = {}
    for row in archived:
        name = row["run_id"]
        source = index[name]
        raw = gzip.decompress((ROOT / source["history"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == row["history_sha256"], name
        points = [r for r in json.loads(raw) if "test_acc" in r and r["step"] <= 100000]
        strict = bool(int(row["strict_grokking"]))
        reached = bool(int(row["test_ever90"]))
        assert audit(points)["audit_fixed_grokking"] == strict
        assert reached == any(r["test_acc"] >= .90 for r in points)
        span = int(row["H"])
        group = 0 if span <= 7 else 1 if span < 15 else 2
        groups[group].append((points, strict, reached))
        histories[name] = {r["step"]: r["test_acc"] for r in points}
    assert len(histories) == len(index) == 192
    assert [len(group) for group in groups] == [60, 72, 60]
    assert [sum(p[1] for p in group) for group in groups] == [0, 32, 0]
    assert [sum(p[2] for p in group) for group in groups] == [0, 49, 57]
    return groups, histories


def matched_local():
    return [
        {"modes": m, "seed": seed, "relay": f"r3_T4_M{m}_seed{seed}",
         "wide": f"r{r}_T{t}_M{m}_seed{seed}"}
        for m, seed, r, t in LOCAL_PAIRS
    ]


def window_pairs():
    base = ROOT / "experiments/transformer_followup_20261115/records"
    pairs = []
    for seed in (0, 1):
        histories = []
        origins = []
        for width in (12, 15):
            run = base / f"T1_W{width}_s{seed}"
            points = json.loads((run / "history.json").read_text())
            histories.append({r["step"]: r["test_acc"] for r in points if "test_acc" in r})
            origins.append(json.loads((run / "origin.json").read_text()))
        keys = ("gpu_uuid", "initial_parameter_hash", "data_sha256", "code_hashes", "torch")
        assert all(origins[0][key] == origins[1][key] for key in keys)
        for h in histories:
            assert set(h).issuperset(range(0, 150001, 500))
        pairs.append((seed, *histories))
    return pairs


def t_sustained(history, threshold):
    return next((s for s in sorted(history) if all(
        history.get(s + delta, -1.) >= threshold for delta in (0, 500, 1000)
    )), None)


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.65)
    ax.spines[["left", "bottom"]].set_color("#667785")
    ax.tick_params(which="major", width=.65, length=2.5, colors=MUTED, pad=2)
    ax.set_axisbelow(True)


def heading(fig, x, y, letter, label):
    fig.text(x, y, letter, fontsize=11.5, fontweight="bold", color=INK, va="center")
    fig.text(x + .027, y, label, fontsize=9.25, fontweight="semibold", color=INK, va="center")


def draw_reach(ax):
    ax.axis("off")
    ax.set(xlim=(-.6, 15.6), ylim=(0, 3.6))
    ax.plot([0, 15], [2.73, 2.73], color=LIGHT, linewidth=1)
    ax.plot([15, 15], [.15, 2.73], color=LIGHT, linewidth=.75, linestyle=THIN)
    ax.scatter([0, 7, 15], [2.73] * 3, color=INK, s=15, zorder=3)
    for x, label, align in [(0, "Input a", "left"), (7, "Input b", "center"), (15, "Readout", "right")]:
        ax.text(x, 2.94, f"{label}\n{x}", ha=align, va="bottom", fontsize=7.25, linespacing=1.0)
    for y, end, label, color, style in (
        (2., 7, r"$H=7$", BASE, "-"),
        (1.19, 12, r"$H=12$", NARROW, "-"),
        (.36, 15, r"$H=15$ or $16$", WIDE, WIDE_DASH),
    ):
        ax.plot([0, end], [y, y], color=color, linestyle=style, linewidth=1.6)
        ax.scatter(end, y, color=color, s=11, zorder=3)
        ax.text(.1, y+.09, label, va="bottom", fontsize=8, color=color)


def draw_scan(ax, group, label):
    axis_style(ax)
    color = NARROW if label == r"$8\leq H\leq14$" else BASE
    for points, strict, _ in group:
        ax.plot([r["step"] / 1000 for r in points], [100 * r["test_acc"] for r in points],
                color=NARROW if strict else BASE, alpha=.29 if strict else .18,
                linewidth=.54)
    grid = sorted({r["step"] for r in group[0][0] if r["step"] % 500 == 0})
    values = [{r["step"]: r["test_acc"] for r in points} for points, _, _ in group]
    medians = [100 * np.median([row[s] for row in values]) for s in grid]
    ax.plot(np.array(grid) / 1000, medians, color=color, linewidth=1.5, zorder=4)
    ax.set(xlim=(0, 100), ylim=(0, 101), xticks=[0, 50, 100], yticks=[0, 50, 100])
    ax.set_title(label + f", n={len(group)}", fontsize=8.0, color=INK, pad=4)
    ax.axhline(90, color=LIGHT, lw=.55, ls=THIN, zorder=0)
    ax.text(.96, .08 if label == r"$H\geq15$" else .91,
            f"Strict grokking: {sum(r[1] for r in group)}",
            transform=ax.transAxes, va="center", ha="right", color=INK, fontsize=7.0,
            bbox=dict(boxstyle="square,pad=.09", facecolor="white", edgecolor="none", alpha=.86),
            zorder=5)


def draw_pair(ax, narrow, wide, label, budget, show_x, show_y):
    axis_style(ax)
    ax.set(xlim=(0, budget / 1000), ylim=(0, 102), xticks=[0, budget / 2000, budget / 1000],
           yticks=[0, 50, 100])
    for threshold in (.1, .9):
        ax.axhline(threshold * 100, color=LIGHT, linewidth=.55, linestyle=THIN)
    for history, color, linestyle, marker in (
        (narrow, NARROW, "-", "o"), (wide, WIDE, WIDE_DASH, "s")
    ):
        times = sorted(s for s in history if s <= budget)
        ax.plot([t/1000 for t in times], [100*history[t] for t in times], linewidth=1.05,
                color=color, linestyle=linestyle, zorder=2)
        for threshold in (.1, .9):
            t = t_sustained(history, threshold)
            if t is not None:
                ax.scatter(t / 1000, 100 * history[t], s=13, facecolor="white", edgecolor=color,
                           linewidth=.8, marker=marker, zorder=5, clip_on=False)
        if t_sustained(history, .9) is None:
            ax.scatter(budget/1000, 100 * history[budget], color=color, s=15, marker=">",
                       zorder=6, clip_on=False)
    ax.set_title(label, fontsize=7.45, color=INK, pad=3.6)
    ax.tick_params(axis="x", labelbottom=show_x)
    ax.tick_params(axis="y", labelleft=show_y)


def paired_effects(narrow, wide):
    first = [t_sustained(trace, .1) for trace in (narrow, wide)]
    last = [t_sustained(trace, .9) for trace in (narrow, wide)]
    assert None not in first
    onset = (first[1] - first[0]) / 1000
    duration = None if None in last else (last[1] - first[1] - last[0] + first[0]) / 1000
    return onset, duration


def row_points(ax, values, position, color, marker, distance):
    lanes = []
    for value in sorted(values):
        for lane in lanes:
            if value - lane[-1] >= distance:
                lane.append(value)
                break
        else:
            lanes.append([value])
    for k, lane in enumerate(lanes):
        y = position + .17 * (k - (len(lanes) - 1) / 2)
        ax.scatter(lane, [y] * len(lane), s=20, marker=marker, c=color, edgecolor="white",
                   linewidth=.55, zorder=4)


def draw_shifts(ax, local, window, bounds, ticks, separation):
    axis_style(ax)
    ax.spines["left"].set_visible(False)
    ax.set(xlim=bounds, ylim=(-.42, 1.42), xticks=ticks, yticks=[1, 0],
           yticklabels=[f"Local ({len(local)})", f"Transformer ({len(window)})"])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.axvline(0, color=LIGHT, linewidth=.83)
    row_points(ax, local, 1, INK, "o", separation)
    row_points(ax, window, 0, ACCENT, "D", separation)
    ax.set_xlabel(r"Wider minus narrower ($10^3$ steps)", labelpad=2, fontsize=8)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "text.color": INK,
        "axes.labelcolor": INK, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.6,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    scans, histories = load_scan()
    local = matched_local()
    window = window_pairs()
    fig = plt.figure(figsize=(7.28, 6.13))

    heading(fig, .075, .966, "A", "Reach within one block")
    heading(fig, .383, .966, "B", "Architecture scan (192 runs)")
    sketch = fig.add_axes([.072, .757, .241, .162])
    draw_reach(sketch)

    scan_axes = fig.add_gridspec(1, 3, left=.398, right=.982, bottom=.754, top=.908, wspace=.20)
    for i, (group, label) in enumerate(zip(scans, (r"$H\leq7$", r"$8\leq H\leq14$", r"$H\geq15$"))):
        ax = fig.add_subplot(scan_axes[0, i])
        draw_scan(ax, group, label)
        ax.tick_params(axis="y", labelleft=i == 0)
        if i == 0:
            ax.set_ylabel("Held-out accuracy (%)", labelpad=1)
    fig.text(.69, .710, r"Training steps ($10^3$)", ha="center", va="center", fontsize=8)

    handles = [
        Line2D([], [], color=NARROW, lw=1.45, label="Narrow (12)"),
        Line2D([], [], color=WIDE, lw=1.45, linestyle=WIDE_DASH, label="Wide (15 or 16)"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.405, .667),
               frameon=False, ncol=2, columnspacing=1.0, handlelength=1.5,
               borderaxespad=0)
    heading(fig, .075, .668, "C", "Local model")
    heading(fig, .765, .668, "D", "Windowed Transformer")

    local_axes = fig.add_gridspec(2, 4, left=.077, right=.723, bottom=.292, top=.609,
                                 hspace=.43, wspace=.26)
    local_effects = []
    for i, pair in enumerate(local):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(local_axes[row, col])
        a, b = histories[pair["relay"]], histories[pair["wide"]]
        label = rf"$M={pair['modes']}$, seed {pair['seed']}"
        draw_pair(ax, a, b, label, 100000, row == 1, col == 0)
        local_effects.append(paired_effects(a, b))
    fig.text(.014, .452, "Held-out accuracy (%)", rotation=90, va="center", fontsize=8)

    window_axes = fig.add_gridspec(2, 1, left=.793, right=.981, bottom=.292, top=.609, hspace=.43)
    window_effects = []
    for i, (seed, a, b) in enumerate(window):
        ax = fig.add_subplot(window_axes[i, 0])
        draw_pair(ax, a, b, f"Seed {seed}", 150000, i == 1, True)
        window_effects.append(paired_effects(a, b))
    fig.text(.51, .246, r"Training steps ($10^3$)", ha="center", va="center", fontsize=8)

    assert [(int(x * 1000), int(y * 1000)) for x, y in window_effects] == [(-77000, 4000), (-64000, 500)]
    assert all(a < 0 and (b is None or b > 0) for a, b in local_effects + window_effects)
    heading(fig, .075, .193, "E", "Onset shift (10%)")
    heading(fig, .552, .193, "F", "Rise duration shift (10–90%)")
    ax = fig.add_axes([.215, .074, .292, .087])
    draw_shifts(ax, [x[0] for x in local_effects], [x[0] for x in window_effects],
                (-86, 5), [-80, -60, -40, -20, 0], 4)
    ax = fig.add_axes([.686, .074, .294, .087])
    draw_shifts(ax, [x[1] for x in local_effects if x[1] is not None], [x[1] for x in window_effects],
                (-3, 46), [0, 10, 20, 30, 40], 2.4)

    path = ROOT / "reproduced" / "communication_span" / "14_communication_span_pairs"
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=270, bbox_inches="tight", pad_inches=.052)
    plt.close(fig)
    complete = sum(duration is not None for _, duration in local_effects)
    assert len(local_effects) == 8 and complete == 6 and len(window_effects) == 2
    print(f"{sum(len(g) for g in scans)} scan runs, {len(local_effects)} local pairs "
          f"({complete} complete rises), and {len(window_effects)} Transformer pairs. Wrote {path}.")


if __name__ == "__main__":
    main()
