# `tables/test-and-build/` — Audit framework source

This directory contains the scripts that regenerate the artifacts in the
parent `tables/` directory from `data/`.

Two kinds of scripts:

| Prefix | Purpose | Output |
|---|---|---|
| `check_T<N>_*.py` | **Test**: recompute paper Table N cell-by-cell and report pass/fail vs. the paper-quoted claims | `../diff_T<N>.csv` + `../table_T<N>_recomputed.csv` |
| `build_subj_acc_tables.py` | **Build**: generate the Sub. Acc. mirror tables for Traps 1 and 2 (these *are* the paper tables, not recomputations of them) | `../table_T_subj_trap{1,2}_final.csv` |

All scripts resolve paths via:

```python
REPO   = Path(__file__).resolve().parent.parent.parent  # repo root
OUTDIR = Path(__file__).resolve().parent.parent         # ../ (tables/)
```

So they can be run from any cwd.

## Special case: `check_T16_app_trap3_full.py`

T16 was authored under the pre-Round-2 *legacy* aggregation rule
(Sub. Acc. = mean across folds, KNN best-HP unlocked) while the rest of the
paper uses the *canonical* rule (median across folds, KNN locked to k=7).
This script supports `--rule {canonical|legacy|both}`. See the script header
for the camera-ready TO DO list.
