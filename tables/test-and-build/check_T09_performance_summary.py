#!/usr/bin/env python3
"""
T9 Sanity Check — Table 9 (tab:performance_summary).

Paper claim (line 458): AD performance summary, 4 rows:
  Row 1 — Subject-dependent 80/20: epoch 83.8% (IQR 8.6 pp) [W_C overlap]
  Row 2 — LPSO P=6, T-test, MLP: subj 83.33% (IQR 16.67 pp), epoch 72.6% (IQR 14.3 pp)
  Row 3 — LPSO P=6, PCA, KNN: subj 50.00% (IQR 33.33 pp), epoch 56.5% (IQR 10.4 pp)
  Row 4 — LPSO P=2, T-test, MLP: subj 100.00% (IQR 50.00 pp), epoch 74.9% (IQR 31.8 pp)

Sources:
  Row 1 — ANOVA_W_C_ad_cntrl_seed{42..51} per-seed accuracies (10 seeds)
  Rows 2-4 — ad_fold_epoch_vs_subject.csv + ad_all_experiments_combined.csv

Tolerance: ±0.5 pp on % values, ±0.5 pp on IQR.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
ROOT = REPO
FOLD_CSV = ROOT/"processed_data/results/ad/ad_fold_epoch_vs_subject.csv"
COMB_CSV = ROOT/"processed_data/results/ad/ad_all_experiments_combined.csv"

PAPER = [
    # (label, subj%, subj_iqr, epoch%, epoch_iqr)
    ("80/20_overlap", None,  None,  83.8, 8.6),
    ("LPSO_P6_Ttest_MLP", 83.33, 16.67, 72.6, 14.3),
    ("LPSO_P6_PCA_KNN",   50.00, 33.33, 56.5, 10.4),
    ("LPSO_P2_Ttest_MLP", 100.00, 50.00, 74.9, 31.8),
]
TOL = 0.5  # pp


def overlap_t_test_stats():
    """Get T-test overlap (W_C) epoch accuracies across 10 seeds × 4 models, AD only."""
    wc = pd.read_csv(REPO / "processed_data/w_c_per_model_accuracies.csv")
    accs = wc[(wc["cohort"] == "ad") & (wc["pipeline"] == "ANOVA")]["accuracy"].tolist()
    if not accs: return None, None
    return np.median(accs)*100, (np.percentile(accs,75)-np.percentile(accs,25))*100


def lpso_stats(P, model, hp_substring, pipe):
    fold = pd.read_csv(FOLD_CSV)
    fold = fold[(fold["pipeline"]==pipe) & (fold["P"]==P) & (fold["model"]==model)]
    fold = fold[fold["hyperparams"].str.contains(hp_substring, na=False, regex=False)]
    if fold.empty: return None, None, None, None
    # Subject acc per fold = mean across (model,hp) which is locked to single hp here
    per_fold_subj = fold.groupby("fold_id")["subject_acc"].mean().values * 100
    s_med = np.median(per_fold_subj)
    s_iqr = np.percentile(per_fold_subj,75) - np.percentile(per_fold_subj,25)
    # Epoch acc per fold
    per_fold_ep = fold.groupby("fold_id")["epoch_acc"].mean().values * 100
    e_med = np.median(per_fold_ep)
    e_iqr = np.percentile(per_fold_ep,75) - np.percentile(per_fold_ep,25)
    return s_med, s_iqr, e_med, e_iqr


def main():
    diff_rows = []
    print(f"{'Row':24s} | {'paper subj/iqr':>16s} {'paper epoch/iqr':>16s} | {'computed':>30s}")
    print("-"*100)

    # Row 1 — overlap
    e_med, e_iqr = overlap_t_test_stats()
    p = PAPER[0]
    print(f"{p[0]:24s} | {'-/-':>16s} {f'{p[3]}/{p[4]}':>16s} | epoch={e_med:.2f}%/iqr={e_iqr:.2f}")
    for cell, claim, comp in [("epoch_med",p[3],e_med),("epoch_iqr",p[4],e_iqr)]:
        d = comp - claim
        stat = "PASS" if abs(d)<=TOL else "FAIL"
        diff_rows.append({"row":p[0],"cell":cell,"claimed":claim,"computed":round(comp,2),"delta":round(d,2),"status":stat,"tolerance":TOL})

    # Row 2 — LPSO P=6 T-test MLP[100]
    s_med, s_iqr, e_med, e_iqr = lpso_stats(6, "MLP", '"hidden_layer_sizes": [100]', "FTest")
    p = PAPER[1]
    print(f"{p[0]:24s} | {f'{p[1]}/{p[2]}':>16s} {f'{p[3]}/{p[4]}':>16s} | subj={s_med:.2f}/{s_iqr:.2f}  ep={e_med:.2f}/{e_iqr:.2f}")
    for cell, claim, comp in [("subj_med",p[1],s_med),("subj_iqr",p[2],s_iqr),("epoch_med",p[3],e_med),("epoch_iqr",p[4],e_iqr)]:
        d = comp - claim
        stat = "PASS" if abs(d)<=TOL else "FAIL"
        diff_rows.append({"row":p[0],"cell":cell,"claimed":claim,"computed":round(comp,2),"delta":round(d,2),"status":stat,"tolerance":TOL})

    # Row 3 — LPSO P=6 PCA KNN k=7
    s_med, s_iqr, e_med, e_iqr = lpso_stats(6, "KNN", '"n_neighbors": 7', "PCA")
    p = PAPER[2]
    print(f"{p[0]:24s} | {f'{p[1]}/{p[2]}':>16s} {f'{p[3]}/{p[4]}':>16s} | subj={s_med:.2f}/{s_iqr:.2f}  ep={e_med:.2f}/{e_iqr:.2f}")
    for cell, claim, comp in [("subj_med",p[1],s_med),("subj_iqr",p[2],s_iqr),("epoch_med",p[3],e_med),("epoch_iqr",p[4],e_iqr)]:
        d = comp - claim
        stat = "PASS" if abs(d)<=TOL else "FAIL"
        diff_rows.append({"row":p[0],"cell":cell,"claimed":claim,"computed":round(comp,2),"delta":round(d,2),"status":stat,"tolerance":TOL})

    # Row 4 — LPSO P=2 T-test MLP[100]
    s_med, s_iqr, e_med, e_iqr = lpso_stats(2, "MLP", '"hidden_layer_sizes": [100]', "FTest")
    p = PAPER[3]
    print(f"{p[0]:24s} | {f'{p[1]}/{p[2]}':>16s} {f'{p[3]}/{p[4]}':>16s} | subj={s_med:.2f}/{s_iqr:.2f}  ep={e_med:.2f}/{e_iqr:.2f}")
    for cell, claim, comp in [("subj_med",p[1],s_med),("subj_iqr",p[2],s_iqr),("epoch_med",p[3],e_med),("epoch_iqr",p[4],e_iqr)]:
        d = comp - claim
        stat = "PASS" if abs(d)<=TOL else "FAIL"
        diff_rows.append({"row":p[0],"cell":cell,"claimed":claim,"computed":round(comp,2),"delta":round(d,2),"status":stat,"tolerance":TOL})

    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUTDIR/"diff_T09.csv", index=False)
    n_pass = (diff["status"]=="PASS").sum()
    print(f"\nT09 RESULT: {n_pass}/{len(diff)} cells PASS")
    if n_pass < len(diff):
        print("\nFailures:")
        print(diff[diff["status"]!="PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
