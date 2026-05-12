# `tables/` — Per-table audit artifacts

This directory contains the **published artifacts** for the paper's table audit:

- `diff_T<N>.csv` — per-cell `(claimed, computed, delta, status, tolerance)`
- `table_T<N>_recomputed.csv` — the full recomputed table
- `table_T_subj_trap{1,2}_final.csv` — Sub. Acc. mirror tables

The code that regenerates these lives in [`test-and-build/`](test-and-build/).
Run a single check or rebuild everything:

```bash
python tables/test-and-build/check_T01_dataset_summary.py
python tables/test-and-build/check_T05_replication_trap1.py
# ... etc
```

Each script reads from `../../data/` (repo `data/`) and writes its artifacts
to the parent `tables/` directory.

## Result summary (latest run)

| Script | Result | Notes |
|---|---|---|
| `check_T01_dataset_summary.py` | 35/35 ✓ | Table 1 — dataset summary |
| `check_T02_fold_designs.py` | 6/6 ✓ | Table 2 — fold counts |
| `check_T04_variance_holdout_AD.py` | 18/18 ✓ | Table 4 — AD holdout variance |
| `check_T05_replication_trap1.py` | 15/15 ✓ | Table 5 — Trap 1 replication |
| `check_T06_replication_trap2.py` | 35/35 ✓ | Table 6 — Trap 2 replication |
| `check_T07_replication_trap3.py` | 15/15 ✓ | Table 7 — Trap 3 replication |
| `check_T09_performance_summary.py` | 14/14 ✓ | Table 9 — performance summary |
| `check_T10_split_manifests.py` | 6/6 ✓ | Table 10 — split-manifest counts |
| `check_T12_biomarkers_AD.py` | 50/50 ✓ | Table 12 — AD biomarkers |
| `check_T13_top_features_per_disease.py` | 24/25 ✓ | Table 13 — top features (one paper-rounding diff: AD rank 5 `O2_Delta` 0.62 vs 0.613) |
| `check_T14_app_trap1_full.py` | 40/40 ✓ | Appendix — Trap 1 full |
| `check_T15_app_trap2_full.py` | 60/60 ✓ | Appendix — Trap 2 full |
| `check_T16_app_trap3_full.py` | 33/120 (canonical), 75/120 (legacy) | **Known divergence — must be updated for camera-ready.** T16 was authored under the pre-Round-2 legacy rule (Sub. Acc. = mean across folds, KNN best-HP unlocked); rest of paper uses the canonical rule (median across folds, KNN locked to k=7). Run with `--rule canonical` for the post-Round-2 numbers (writes `table_T16_canonical.csv` — use these for the camera-ready); `--rule legacy` reproduces the submitted version; `--rule both` shows both. See script header for the camera-ready TO DO list. |

`build_subj_acc_tables.py` recomputes the "Sub. Acc." mirror tables for Traps 1 and 2.

## Data dependencies

All scripts read only from `data/` in this repo. In particular, the four scripts
that originally walked ~930 MB of `ml_results_grid_search/` raw outputs now read
two small pre-computed summary CSVs:

- `data/w_c_per_model_accuracies.csv` — per `(cohort, pipeline, seed, model)` accuracy
- `data/w_c_subject_acc.csv` — per `(cohort, pipeline, seed, model, task)` Sub. Acc. (majority vote)

These are produced offline by `tools/precompute_wc_summary.py` and
`tools/precompute_wc_subj_acc.py` from the raw pipeline outputs (not shipped in
this repo). Set the `SRC` environment variable to your local pipeline-output
tree before running:

```bash
SRC=/your/path/to/eeg-pipeline-outputs python tools/precompute_wc_summary.py
SRC=/your/path/to/eeg-pipeline-outputs python tools/precompute_wc_subj_acc.py
```
