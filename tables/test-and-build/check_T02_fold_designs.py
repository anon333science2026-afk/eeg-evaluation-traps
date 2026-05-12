#!/usr/bin/env python3
"""
T2 Sanity Check — Table 2 (tab:fold_designs) — Experiment configurations.

Paper claims:
  • LPSO P=6  : 50 unique folds (T-test pipeline + PCA pipeline)
  • LPSO P=2  : 50 unique folds (T-test pipeline + PCA pipeline)
  • SO        : 10 seeds (T-test + PCA)

Verification:
  • Count unique fold_id values per (pipeline, P) in <dis>_all_experiments_combined.csv
  • Count seed-named directories for SO (W_C) experiments
  • Use AD as the canonical cohort for the structural check (paper Table 2 isn't
    cohort-specific — same design across all 5 datasets).

Tolerance: exact integer counts.
"""
import pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
ROOT = REPO
COMB_CSV = ROOT / "data/results/ad/ad_all_experiments_combined.csv"

# Paper claims: (experiment_id, pipeline, split_type, P, unique_folds)
PAPER = [
    # (experiment_id, feature_set_in_csv, split_type, P, paper_n_folds)
    ("T-test_P_6", "ANOVA", "LPSO", 6, 50),
    ("PCA_P_6",    "PCA",   "LPSO", 6, 50),
    ("T-test_P_2", "ANOVA", "LPSO", 2, 50),
    ("PCA_P_2",    "PCA",   "LPSO", 2, 50),
    ("T-test_SO",  "ANOVA", "SO",   None, 10),
    ("PCA_SO",     "PCA",   "SO",   None, 10),
]


def main():
    df = pd.read_csv(COMB_CSV)
    diff_rows = []
    table_rows = []
    print(f"{'Experiment':14s} {'feat_set':9s} {'Split':6s} {'P':>4s} | {'paper':>6s} {'computed':>10s} status")
    print("-" * 70)
    for exp_id, pipe, split, P, paper_n in PAPER:
        if split == "LPSO":
            sub = df[(df["feature_set"] == pipe) & (df["holdout_size_P"] == P) & (df["experiment_type"] == "LPSO_Random_50")]
            n_folds = sub["fold_id"].nunique()
            note = f"unique fold_id where P={P}, feature_set={pipe}"
        else:  # SO — count unique seeds present in W_C summary CSV (AD cohort)
            pipe_key = "ANOVA" if pipe == "FTest" else "PCA"
            wc = pd.read_csv(ROOT / "data/w_c_per_model_accuracies.csv")
            n_folds = wc[(wc["cohort"] == "ad") & (wc["pipeline"] == pipe_key)]["seed"].nunique()
            note = f"unique W_C seeds in data/w_c_per_model_accuracies.csv (cohort=ad, pipeline={pipe_key})"

        delta = n_folds - paper_n
        stat = "PASS" if delta == 0 else "FAIL"
        diff_rows.append({"experiment": exp_id, "claimed": paper_n, "computed": n_folds, "delta": delta, "status": stat, "method": note})
        table_rows.append({"experiment": exp_id, "pipeline": pipe, "split": split, "P": P, "unique_folds": n_folds})
        print(f"{exp_id:14s} {pipe:9s} {split:6s} {str(P):>4s} | {paper_n:>6d} {n_folds:>10d} {'✓' if stat=='PASS' else '✗'}")

    diff = pd.DataFrame(diff_rows)
    out_diff = OUTDIR / "diff_T02.csv"
    out_table = OUTDIR / "table_T02_recomputed.csv"
    diff.to_csv(out_diff, index=False)
    pd.DataFrame(table_rows).to_csv(out_table, index=False)
    n_pass = (diff["status"] == "PASS").sum()
    print(f"\nT02 RESULT: {n_pass}/{len(diff)} cells PASS")
    print(f"  Diff: {out_diff.name}  Table: {out_table.name}")
    if n_pass < len(diff):
        print("\nFailures:")
        print(diff[diff["status"] != "PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
