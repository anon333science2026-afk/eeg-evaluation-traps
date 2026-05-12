#!/usr/bin/env python3
"""
T7 Sanity Check — Table 7 (tab:replication_trap3) — Trap 3 across diseases.

Paper post-restructure (line 340):
  | Dataset | Avg Ep. Acc. | Avg Sub. Acc. | Avg Gap (pp) |

Aggregation rule (canonical):
  best HP per model (by median test_accuracy) → MEDIAN across 50 folds → mean of 4 models
  Both epoch_acc and subject_acc collected from <dis>_fold_epoch_vs_subject.csv
  for the best-HP-per-model identified in <dis>_all_experiments_combined.csv.

Tolerance: ±0.005 on accuracies, ±0.2 pp on gap.
"""
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
KNN_K7 = '"n_neighbors": 7'
MODELS = ["MLP","XGBoost","SVM","KNN"]

DATASETS = [
    # (name, fold_csv, fold_pipe_col, fold_pipe_val, condition, combined_csv, combined_exp_name)
    ("AD",     f"{REPO}/data/results/ad/ad_fold_epoch_vs_subject.csv",   "pipeline","FTest", None,
               f"{REPO}/data/results/ad/ad_all_experiments_combined.csv", "ANOVA_L_6_Random"),
    ("FTD",    f"{REPO}/data/results/ftd/ftd_fold_epoch_vs_subject.csv",     "pipeline","FTest", None,
               f"{REPO}/data/results/ftd/ftd_all_experiments_combined.csv", "ANOVA_L_6_FTD"),
    ("MDD EC", f"{REPO}/data/results/mdd/mdd_fold_epoch_vs_subject.csv", "feature_set","ANOVA","EC",
               f"{REPO}/data/results/mdd/mdd_all_experiments_combined.csv", "ANOVA_L_6_EC"),
    ("MDD EO", f"{REPO}/data/results/mdd/mdd_fold_epoch_vs_subject.csv", "feature_set","ANOVA","EO",
               f"{REPO}/data/results/mdd/mdd_all_experiments_combined.csv", "ANOVA_L_6_EO"),
    ("SCZ",    f"{REPO}/data/results/scz/scz_fold_epoch_vs_subject.csv", "pipeline","FTest", None,
               f"{REPO}/data/results/scz/scz_all_experiments_combined.csv", "ANOVA_L_6_SCZ"),
]
PAPER = {"AD":(0.715,0.833,11.9),"FTD":(0.627,0.667,3.9),"MDD EC":(0.814,0.854,4.1),
         "MDD EO":(0.781,0.833,5.2),"SCZ":(0.573,0.667,9.4)}
TOL = dict(epoch=0.005, subj=0.005, gap=0.2)


def main():
    diff_rows = []
    table_rows = []
    print(f"{'D':6s} | {'Paper':<22s} | {'Computed':<22s} | status")
    print("-"*70)
    for name, fp, pcol, pval, cond, cp, exp in DATASETS:
        comb = pd.read_csv(cp)
        fold = pd.read_csv(fp)
        fold = fold[fold[pcol]==pval]
        if cond and "condition" in fold.columns:
            fold = fold[fold["condition"]==cond]
        fold = fold[fold["P"]==6]

        e_meds, s_meds = [], []
        for model in MODELS:
            if model == "KNN":
                best_hp = '{"metric": "euclidean", "n_neighbors": 7, "weights": "uniform"}'
            else:
                m = comb[(comb["experiment"]==exp)&(comb["model"]==model)]
                if m.empty: continue
                best_hp = m.groupby("hyperparams")["test_accuracy"].median().idxmax()
            sub = fold[(fold["model"]==model)&(fold["hyperparams"]==best_hp)]
            if sub.empty: continue
            e_meds.append(sub["epoch_acc"].median())
            s_meds.append(sub["subject_acc"].median())
        avg_e = float(np.mean(e_meds))
        avg_s = float(np.mean(s_meds))
        gap_pp = (avg_s - avg_e) * 100

        pe, ps, pg = PAPER[name]
        table_rows.append({"dataset":name,"avg_Ep_Acc":round(avg_e,4),"avg_Sub_Acc":round(avg_s,4),"avg_gap_pp":round(gap_pp,2)})
        for cell, claim, comp, tol in [("epoch",pe,avg_e,TOL["epoch"]),
                                         ("subj",ps,avg_s,TOL["subj"]),
                                         ("gap_pp",pg,gap_pp,TOL["gap"])]:
            d = comp - claim
            stat = "PASS" if abs(d) <= tol else "FAIL"
            diff_rows.append({"dataset":name,"cell":cell,"claimed":claim,"computed":round(comp,4),"delta":round(d,4),"status":stat,"tolerance":tol})
        ok = all(d["status"]=="PASS" for d in diff_rows[-3:])
        print(f"{name:6s} | {f'{pe:.3f} / {ps:.3f} / {pg:+.1f}':<22s} | {f'{avg_e:.3f} / {avg_s:.3f} / {gap_pp:+.1f}':<22s} | {'✓' if ok else '✗'}")

    diff = pd.DataFrame(diff_rows)
    out_diff = OUTDIR / "diff_T07.csv"
    out_table = OUTDIR / "table_T07_recomputed.csv"
    diff.to_csv(out_diff, index=False)
    pd.DataFrame(table_rows).to_csv(out_table, index=False)
    n_pass = (diff["status"]=="PASS").sum()
    print(f"\nT07 RESULT: {n_pass}/{len(diff)} cells PASS")
    if n_pass < len(diff):
        print("\nFailures:")
        print(diff[diff["status"]!="PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
