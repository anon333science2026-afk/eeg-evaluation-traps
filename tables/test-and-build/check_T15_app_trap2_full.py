#!/usr/bin/env python3
"""
T15 Sanity Check — Table 15 (tab:app_trap2_full) — Appendix B.

Paper structure: per disease × {T-test, PCA} pipeline:
  | T-test P=6 IQR Min/Max | T-test P=2 IQR Min/Max | T-test ratio Min/Max |
  | PCA P=6 IQR Min/Max | PCA P=2 IQR Min/Max | PCA ratio Min/Max |

Per-model IQR = IQR of best-HP test_accuracy across 50 folds (in pp).
Min/Max columns = range over the 4 model IQRs.
Ratio = per-model (P=2 IQR / P=6 IQR), then min/max across 4 models.

Tolerance: ±0.5 pp on IQR, ±0.2× on ratio.
"""
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
ROOT = REPO
KNN_K7 = '"n_neighbors": 7'
MODELS = ["MLP","XGBoost","SVM","KNN"]

# Paper claims (lines 749-753)
PAPER = {
    # disease: (T_p6_min, T_p6_max, T_p2_min, T_p2_max, T_r_min, T_r_max,
    #          P_p6_min, P_p6_max, P_p2_min, P_p2_max, P_r_min, P_r_max)
    "AD":     ( 9.0, 14.3, 19.2, 32.1, 1.8, 2.7,  6.5, 10.4,  8.4, 15.0, 0.8, 1.9),
    "FTD":    ( 9.2, 13.7, 13.5, 24.4, 1.5, 2.0,  4.2,  5.4,  7.7,  7.8, 1.4, 1.9),
    "MDD EC": (10.5, 17.8, 25.4, 34.9, 1.7, 2.5,  1.8, 17.8,  1.0, 39.4, 0.6, 2.2),
    "MDD EO": (13.4, 15.4, 14.8, 47.2, 1.1, 3.1,  0.6, 10.1,  0.6, 42.0, 1.0, 4.2),
    "SCZ":    ( 7.2,  9.5, 14.7, 24.0, 1.7, 2.5,  2.9, 12.8,  7.0, 30.7, 0.9, 2.4),
}
TOL = dict(iqr=0.5, ratio=0.2)

COMB_CSV = {
    "AD":     ROOT/"data/results/ad/ad_all_experiments_combined.csv",
    "FTD":    ROOT/"data/results/ftd/ftd_all_experiments_combined.csv",
    "MDD EC": ROOT/"data/results/mdd/mdd_all_experiments_combined.csv",
    "MDD EO": ROOT/"data/results/mdd/mdd_all_experiments_combined.csv",
    "SCZ":    ROOT/"data/results/scz/scz_all_experiments_combined.csv",
}
EXP = {
    "AD":     dict(t6="ANOVA_L_6_Random", t2="ANOVA_L_2_Random", p6="PCA_L_6_Random", p2="PCA_L_2_Random"),
    "FTD":    dict(t6="ANOVA_L_6_FTD",    t2="ANOVA_L_2_FTD",    p6="PCA_L_6_FTD",    p2="PCA_L_2_FTD"),
    "MDD EC": dict(t6="ANOVA_L_6_EC",     t2="ANOVA_L_2_EC",     p6="PCA_L_6_EC",     p2="PCA_L_2_EC"),
    "MDD EO": dict(t6="ANOVA_L_6_EO",     t2="ANOVA_L_2_EO",     p6="PCA_L_6_EO",     p2="PCA_L_2_EO"),
    "SCZ":    dict(t6="ANOVA_L_6_SCZ",    t2="ANOVA_L_2_SCZ",    p6="PCA_L_6_SCZ",    p2="PCA_L_2_SCZ"),
}


def best_hp_iqr(comb, exp_name):
    rows = {}
    for m in MODELS:
        sub = comb[(comb["experiment"]==exp_name) & (comb["model"]==m)]
        if sub.empty: continue
        if m == "KNN":
            sk = sub[sub["hyperparams"].str.contains(KNN_K7, na=False)]
            if sk.empty: continue
            accs = sk["test_accuracy"].values
        else:
            best = sub.groupby("hyperparams")["test_accuracy"].median().idxmax()
            accs = sub[sub["hyperparams"]==best]["test_accuracy"].values
        if len(accs)==0: continue
        q1, q3 = np.percentile(accs, [25, 75])
        rows[m] = (q3 - q1) * 100
    return rows


def main():
    diff_rows, table_rows = [], []
    print(f"{'D':6s} | T-IQR-P6 / T-IQR-P2 / ratio | P-IQR-P6 / P-IQR-P2 / ratio")
    print("-" * 95)
    for name in PAPER:
        comb = pd.read_csv(COMB_CSV[name])
        t6 = best_hp_iqr(comb, EXP[name]["t6"])
        t2 = best_hp_iqr(comb, EXP[name]["t2"])
        p6 = best_hp_iqr(comb, EXP[name]["p6"])
        p2 = best_hp_iqr(comb, EXP[name]["p2"])

        common_t = [m for m in MODELS if m in t6 and m in t2]
        common_p = [m for m in MODELS if m in p6 and m in p2]

        t_p6_min, t_p6_max = min(t6.values()), max(t6.values())
        t_p2_min, t_p2_max = min(t2.values()), max(t2.values())
        t_ratios = [t2[m]/t6[m] for m in common_t if t6[m] > 0]
        t_r_min, t_r_max = min(t_ratios), max(t_ratios)

        p_p6_min, p_p6_max = min(p6.values()), max(p6.values())
        p_p2_min, p_p2_max = min(p2.values()), max(p2.values())
        p_ratios = [p2[m]/p6[m] for m in common_p if p6[m] > 0]
        p_r_min, p_r_max = min(p_ratios), max(p_ratios)

        p = PAPER[name]
        table_rows.append({"disease":name,
                           "T_p6_min":round(t_p6_min,1),"T_p6_max":round(t_p6_max,1),
                           "T_p2_min":round(t_p2_min,1),"T_p2_max":round(t_p2_max,1),
                           "T_r_min":round(t_r_min,2),"T_r_max":round(t_r_max,2),
                           "P_p6_min":round(p_p6_min,1),"P_p6_max":round(p_p6_max,1),
                           "P_p2_min":round(p_p2_min,1),"P_p2_max":round(p_p2_max,1),
                           "P_r_min":round(p_r_min,2),"P_r_max":round(p_r_max,2)})
        cells = [
            ("T_p6_min",p[0],t_p6_min,TOL["iqr"]),("T_p6_max",p[1],t_p6_max,TOL["iqr"]),
            ("T_p2_min",p[2],t_p2_min,TOL["iqr"]),("T_p2_max",p[3],t_p2_max,TOL["iqr"]),
            ("T_r_min",p[4],t_r_min,TOL["ratio"]),("T_r_max",p[5],t_r_max,TOL["ratio"]),
            ("P_p6_min",p[6],p_p6_min,TOL["iqr"]),("P_p6_max",p[7],p_p6_max,TOL["iqr"]),
            ("P_p2_min",p[8],p_p2_min,TOL["iqr"]),("P_p2_max",p[9],p_p2_max,TOL["iqr"]),
            ("P_r_min",p[10],p_r_min,TOL["ratio"]),("P_r_max",p[11],p_r_max,TOL["ratio"]),
        ]
        for cell, claim, comp, tol in cells:
            d = comp - claim
            stat = "PASS" if abs(d) <= tol else "FAIL"
            diff_rows.append({"disease":name,"cell":cell,"claimed":claim,"computed":round(comp,3),"delta":round(d,3),"status":stat,"tolerance":tol})
        print(f"{name:6s} | {t_p6_min:.1f}-{t_p6_max:.1f}/{t_p2_min:.1f}-{t_p2_max:.1f}/{t_r_min:.1f}-{t_r_max:.1f}x | {p_p6_min:.1f}-{p_p6_max:.1f}/{p_p2_min:.1f}-{p_p2_max:.1f}/{p_r_min:.1f}-{p_r_max:.1f}x")

    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUTDIR/"diff_T15.csv", index=False)
    pd.DataFrame(table_rows).to_csv(OUTDIR/"table_T15_recomputed.csv", index=False)
    n_pass = (diff["status"]=="PASS").sum()
    print(f"\nT15 RESULT: {n_pass}/{len(diff)} cells PASS")
    if n_pass < len(diff):
        print(diff[diff["status"]!="PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
