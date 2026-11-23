# Frozen-checkpoint evidence for TGLT

This release contains the reported evaluations, figure, and analysis code. Review decisions and claim scope are in [the evidence review](PNAS证据闭环.md).

## Replot from recorded results

Run `python figure.py` from any directory in an environment with NumPy and Matplotlib. This uses `summary.json` and no model weights. `python summarize.py` first rebuilds that summary from the CSV and gzip files and checks the rosters and accounting identities.

## Rerun the checkpoint analysis

Use Python 3.8.20 and PyTorch 2.4.1 with the TGLT research archive. Set `TGLT_SOURCE_ROOT` to the archive root, whose Git checkout is `TGLT-Grokking/`. The checkout used for the replay was `36d12468f8f2bc4ebc95550ce6cd07c25520af8c`. The archive supplies the frozen checkpoints, split tables, and historical controls under these directories.

- `analysis/firstgen_beta_wd_phase_20261027/runs/` and `analysis/firstgen_beta_wd_gray_continuation_20261102/continuation_300k/` for the three H12 add trajectories
- `experiments/firstgen_followups_20261114/` for the moved-input add task
- `week/v2/code/` and `week/v2/results/` for subtraction and multiplication
- `week/v1/` and `releases/TGLT_mechanism_data_20261111/` for the existing model and data helpers

Run `python round1.py`, `python round2.py geometry`, `python round2.py temporal`, `python round3.py`, `python summarize.py`, and `python figure.py` in an analysis copy when replaying. The output tables record the checkpoint hashes and the counts from the historical training logs. The CPU replay differences of one answer at two checkpoints are retained.

The seeds are the comparison units. The eight-offset comparisons, 58-offset sensitivity checks, repeated saved states, and test pairs are within-run measurements. Figure panels a and b use the mean across the full 58 wrong pairings and feature-centered norm matching. Panel d uses the eight recorded offsets and the actual saved steps listed in `temporal_anchors.json`.

Raw weights are kept in the research archive. The requested public publication archive and checkpoint access arrangements should be stated in the paper's data availability plan.
