#!/usr/bin/env python3
"""
T6 Sanity Check — Table 6 (tab:replication_trap2) — NEW 7-COLUMN STRUCTURE 2026-05-07.

Paper post-restructure (line 322–331):
  | Dataset | Ep.Acc. IQR P=6 (pp) | Ep.Acc. IQR P=2 (pp) | IQR ratio
  | Max Ep.Acc. (P=6) | Max Ep.Acc. (P=2) | Max Sub.Acc. (P=6) | Max Sub.Acc. (P=2) |

Aggregation rule (canonical, locked Round 2):
  best HP per model (by median test_accuracy)
  → IQR / max are computed on those 50 best-HP fold accuracies
  → IQR averaged across 4 models; max taken across 4 models

Sub. Acc. side: max single-fold subject_acc from <dis>_fold_epoch_vs_subject.csv,
across all (fold, model, hyperparam) — represents the rhetorical "lucky fold" claim.

Tolerance: ±0.5 pp on IQR, ±0.1× on ratio, ±0.005 on max accuracies.
"""
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
KNN_K7 = '"n_neighbors": 7'
MODELS = ["MLP","XGBoost","SVM","KNN"]

DATASETS = [
    ("AD",     f"{REPO}/data/results/ad/ad_all_experiments_combined.csv",         "ANOVA_L_6_Random",  "ANOVA_L_2_Random",
               f"{REPO}/data/results/ad/ad_fold_epoch_vs_subject.csv", "pipeline","FTest", None),
    ("FTD",    f"{REPO}/data/results/ftd/ftd_all_experiments_combined.csv",            "ANOVA_L_6_FTD",     "ANOVA_L_2_FTD",
               f"{REPO}/data/results/ftd/ftd_fold_epoch_vs_subject.csv", "pipeline","FTest", None),
    ("MDD EC", f"{REPO}/data/results/mdd/mdd_all_experiments_combined.csv",        "ANOVA_L_6_EC",      "ANOVA_L_2_EC",
               f"{REPO}/data/results/mdd/mdd_fold_epoch_vs_subject.csv", "feature_set","ANOVA","EC"),
    ("MDD EO", f"{REPO}/data/results/mdd/mdd_all_experiments_combined.csv",        "ANOVA_L_6_EO",      "ANOVA_L_2_EO",
               f"{REPO}/data/results/mdd/mdd_fold_epoch_vs_subject.csv", "feature_set","ANOVA","EO"),
    ("SCZ",    f"{REPO}/data/results/scz/scz_all_experiments_combined.csv", "ANOVA_L_6_SCZ",     "ANOVA_L_2_SCZ",
               f"{REPO}/data/results/scz/scz_fold_epoch_vs_subject.csv", "pipeline","FTest", None),
]
PAPER = {  # iqr_p6, iqr_p2, ratio, ep_p6, ep_p2, sub_p6, sub_p2
    "AD":     (11.8, 26.2, 2.2, 0.924, 0.975, 1.000, 1.000),
    "FTD":    (11.3, 20.4, 1.8, 0.856, 0.973, 1.000, 1.000),
    "MDD EC": (14.1, 30.6, 2.2, 0.996, 1.000, 1.000, 1.000),
    "MDD EO": (13.8, 26.8, 1.9, 0.989, 1.000, 1.000, 1.000),
    "SCZ":    ( 8.7, 17.9, 2.1, 0.788, 0.834, 1.000, 1.000),
}
TOL = dict(iqr=0.5, ratio=0.1, acc=0.005)


def best_hp_iqr_max(comb, exp_name):
    rows = []
    for model in MODELS:
        m = comb[(comb["experiment"]==exp_name) & (comb["model"]==model)]
        if m.empty: continue
        if model == "KNN":
            mk = m[m["hyperparams"].str.contains(KNN_K7, na=False)]
            if mk.empty: continue
            accs = mk["test_accuracy"].values
        else:
            best_hp = m.groupby("hyperparams")["test_accuracy"].median().idxmax()
            accs = m[m["hyperparams"]==best_hp]["test_accuracy"].values
        if len(accs)==0: continue
        q1, q3 = np.percentile(accs, [25, 75])
        rows.append((model, (q3-q1)*100, accs.max()))
    return rows


def max_subj_acc_per_fold(fold_df, P):
    f = fold_df[fold_df["P"]==P]
    if f.empty: return None
    per_fold = f.groupby(["fold_id","model","hyperparams"])["subject_acc"].mean()
    return per_fold.max()


def main():
    diff_rows = []
    table_rows = []
    print(f"{'D':6s} | {'iqr_p6':>9s} {'iqr_p2':>9s} {'ratio':>7s} | {'ep_p6':>8s} {'ep_p2':>8s} | {'sub_p6':>8s} {'sub_p2':>8s}")
    print("-"*90)
    for name, ccsv, e6, e2, fcsv, fpcol, fpval, fcond in DATASETS:
        comb = pd.read_csv(ccsv)
        fold = pd.read_csv(fcsv)
        fold = fold[fold[fpcol]==fpval]
        if fcond and "condition" in fold.columns:
            fold = fold[fold["condition"]==fcond]

        rows6 = best_hp_iqr_max(comb, e6)
        rows2 = best_hp_iqr_max(comb, e2)
        avg_iqr6 = float(np.mean([r[1] for r in rows6]))
        avg_iqr2 = float(np.mean([r[1] for r in rows2]))
        ratio = avg_iqr2 / avg_iqr6 if avg_iqr6 > 0 else float("nan")
        max_ep_p6 = max(r[2] for r in rows6)
        max_ep_p2 = max(r[2] for r in rows2)
        max_sub_p6 = max_subj_acc_per_fold(fold, 6)
        max_sub_p2 = max_subj_acc_per_fold(fold, 2)

        p_iqr6, p_iqr2, p_ratio, p_ep6, p_ep2, p_sub6, p_sub2 = PAPER[name]
        table_rows.append({"dataset":name,
                           "iqr_p6_pp":round(avg_iqr6,2),"iqr_p2_pp":round(avg_iqr2,2),"ratio":round(ratio,2),
                           "max_ep_p6":round(max_ep_p6,4),"max_ep_p2":round(max_ep_p2,4),
                           "max_sub_p6":round(max_sub_p6,4),"max_sub_p2":round(max_sub_p2,4)})

        cells = [("iqr_p6", p_iqr6, avg_iqr6, TOL["iqr"]),
                 ("iqr_p2", p_iqr2, avg_iqr2, TOL["iqr"]),
                 ("ratio",  p_ratio, ratio,    TOL["ratio"]),
                 ("max_ep_p6", p_ep6, max_ep_p6, TOL["acc"]),
                 ("max_ep_p2", p_ep2, max_ep_p2, TOL["acc"]),
                 ("max_sub_p6", p_sub6, max_sub_p6, TOL["acc"]),
                 ("max_sub_p2", p_sub2, max_sub_p2, TOL["acc"])]
        for cell, claim, comp, tol in cells:
            d = comp - claim
            stat = "PASS" if abs(d) <= tol else "FAIL"
            diff_rows.append({"dataset":name,"cell":cell,"claimed":claim,"computed":round(comp,4),"delta":round(d,4),"status":stat,"tolerance":tol})

        def fmt(p, c, t): return f"{c:>7.2f}{'✓' if abs(c-p)<=t else '✗'}"
        print(f"{name:6s} | {fmt(p_iqr6,avg_iqr6,TOL['iqr']):>9s} {fmt(p_iqr2,avg_iqr2,TOL['iqr']):>9s} {ratio:>5.1f}x{'✓' if abs(ratio-p_ratio)<=TOL['ratio'] else '✗'} | "
              f"{fmt(p_ep6,max_ep_p6,TOL['acc']):>7s} {fmt(p_ep2,max_ep_p2,TOL['acc']):>7s} | "
              f"{fmt(p_sub6,max_sub_p6,TOL['acc']):>7s} {fmt(p_sub2,max_sub_p2,TOL['acc']):>7s}")

    diff = pd.DataFrame(diff_rows)
    out_diff = OUTDIR / "diff_T06.csv"
    out_table = OUTDIR / "table_T06_recomputed.csv"
    diff.to_csv(out_diff, index=False)
    pd.DataFrame(table_rows).to_csv(out_table, index=False)
    n_pass = (diff["status"]=="PASS").sum()
    print(f"\nT06 RESULT: {n_pass}/{len(diff)} cells PASS")
    if n_pass < len(diff):
        print("\nFailures:")
        print(diff[diff["status"]!="PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
