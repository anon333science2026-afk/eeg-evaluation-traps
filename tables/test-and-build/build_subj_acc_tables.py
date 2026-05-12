#!/usr/bin/env python3
"""
Build Sub. Acc. mirror tables for Trap 1 (tab:app_subj_trap1) and Trap 2
(tab:app_subj_trap2). Computes from canonical sources:

  Disjoint subject_acc → <dis>_fold_epoch_vs_subject.csv (already pre-computed
                          per-fold per-(model, hp))
  Overlap subject_acc  → derived from
                          <prefix>_seed{42..51}/ml_results_grid_search/<model>/
                              within_subject_split/task_*/test_predictions.parquet
                          via majority vote per subject

For consistency across cohorts: capped at first 3 task dirs per (seed, model)
since SCZ seed 42 has 36 task dirs (re-runs) while all other (seed, disease)
combinations have exactly 12 (4 models × 3 HPs).

Outputs:
  table_T_subj_trap1_final.csv  — Trap 1 Sub. Acc. (5 rows)
  table_T_subj_trap2_final.csv  — Trap 2 Sub. Acc. (5 rows, all metrics)
"""
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

ROOT = REPO
OUTDIR = Path(__file__).resolve().parent.parent

DJ = [
    ("AD",     "data/results/ad/ad_fold_epoch_vs_subject.csv",   "pipeline","FTest", None),
    ("FTD",    "data/results/ftd/ftd_fold_epoch_vs_subject.csv",     "pipeline","FTest", None),
    ("MDD-EC", "data/results/mdd/mdd_fold_epoch_vs_subject.csv", "feature_set","ANOVA","EC"),
    ("MDD-EO", "data/results/mdd/mdd_fold_epoch_vs_subject.csv", "feature_set","ANOVA","EO"),
    ("SCZ",    "data/results/scz/scz_fold_epoch_vs_subject.csv", "pipeline","FTest", None),
]
WC = {
    "AD":     ("ANOVA_W_C_ad_cntrl_seed{seed}", ""),
    "FTD":    ("ANOVA_W_C_ftd_vs_cntrl_seed{seed}", "data/ftd_vs_C/intra-subject_seed_runs"),
    "MDD-EC": ("ANOVA_W_C_mdd_cs_cntrl_seed{seed}_EC", "data/mdd_vs_cntrl"),
    "MDD-EO": ("ANOVA_W_C_mdd_cs_cntrl_seed{seed}_eyesopen_EO", "data/mdd_vs_cntrl"),
    "SCZ":    ("ANOVA_W_C_scz_cntrl_seed{seed}", "scz_vs_cntrl_all_results"),
}


_WC_SUBJ_CACHE = None
_COHORT_KEY = {"AD": "ad", "FTD": "ftd", "MDD-EC": "mdd_ec", "MDD-EO": "mdd_eo", "SCZ": "scz"}

def overlap_subj(cohort_name, pipeline="ANOVA"):
    """Lookup pre-computed Sub. Acc. per (seed, model, task) from
    data/w_c_subject_acc.csv (built by tools/precompute_wc_subj_acc.py).
    """
    global _WC_SUBJ_CACHE
    if _WC_SUBJ_CACHE is None:
        _WC_SUBJ_CACHE = pd.read_csv(REPO / "data/w_c_subject_acc.csv")
    df = _WC_SUBJ_CACHE
    return df[(df["cohort"] == _COHORT_KEY[cohort_name]) & (df["pipeline"] == pipeline)]


def main():
    # Trap 1
    print("TRAP 1 — Sub. Acc.")
    print(f"{'Disease':8s} | {'Disj P=6':>9s} {'Overlap':>8s} {'Inflation':>10s}")
    print("-" * 50)
    trap1 = []
    for name, dj_path, pcol, pval, cond in DJ:
        df = pd.read_csv(ROOT / dj_path)
        df = df[df[pcol] == pval]
        if cond and "condition" in df.columns:
            df = df[df["condition"] == cond]
        d6 = df[df["P"] == 6]
        disj = d6["subject_acc"].mean()
        ov_df = overlap_subj(name, "ANOVA")
        ov = ov_df["subject_acc"].mean()
        infl = (ov - disj) * 100
        trap1.append({"disease": name, "disj_p6": round(disj, 4),
                       "overlap": round(ov, 4), "inflation_pp": round(infl, 2)})
        print(f"{name:8s} | {disj:>9.4f} {ov:>8.4f} {infl:>+9.1f}")

    # Trap 2
    print(f"\nTRAP 2 — Sub. Acc.")
    print(f"{'Disease':8s} | {'mean P=6':>9s} {'mean P=2':>9s} {'IQR P=6':>8s} {'IQR P=2':>8s} {'ratio':>6s} {'%lucky 6':>9s} {'%lucky 2':>9s}")
    print("-" * 95)
    trap2 = []
    for name, dj_path, pcol, pval, cond in DJ:
        row = {"disease": name}
        for P in [6, 2]:
            df = pd.read_csv(ROOT / dj_path)
            df = df[df[pcol] == pval]
            if cond and "condition" in df.columns:
                df = df[df["condition"] == cond]
            df = df[df["P"] == P]
            per_fold = df.groupby("fold_id")["subject_acc"].mean()
            any_at_1 = df.groupby("fold_id")["subject_acc"].max() >= 0.999
            q1, q3 = per_fold.quantile([0.25, 0.75])
            suf = f"_p{P}"
            row[f"mean{suf}"] = round(df["subject_acc"].mean(), 4)
            row[f"iqr{suf}_pp"] = round((q3 - q1) * 100, 2)
            row[f"pct_lucky{suf}"] = round(100 * any_at_1.mean(), 1)
        row["iqr_ratio"] = round(row["iqr_p2_pp"] / row["iqr_p6_pp"], 2) if row["iqr_p6_pp"] > 0 else float("nan")
        trap2.append(row)
        print(f"{name:8s} | {row['mean_p6']:>9.4f} {row['mean_p2']:>9.4f} {row['iqr_p6_pp']:>7.1f}p {row['iqr_p2_pp']:>7.1f}p {row['iqr_ratio']:>5.1f}x {row['pct_lucky_p6']:>8.1f}% {row['pct_lucky_p2']:>8.1f}%")

    pd.DataFrame(trap1).to_csv(OUTDIR / "table_T_subj_trap1_final.csv", index=False)
    pd.DataFrame(trap2).to_csv(OUTDIR / "table_T_subj_trap2_final.csv", index=False)
    print(f"\nSaved → table_T_subj_trap1_final.csv, table_T_subj_trap2_final.csv")


if __name__ == "__main__":
    main()
