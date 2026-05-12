#!/usr/bin/env python3
"""
T4 Sanity Check — Table 4 (tab:variance_holdout) — AD Trap 2 exemplar IQR cells.

Paper claim: 4 specific (Feature, P, Model-HP) cells with median, IQR, min, max,
ratio. Source: ad_all_experiments_combined.csv.

  Row 1: T-test, P=6, MLP layers=[100]              → 72.6%, IQR 14.3, 54.9–90.0
  Row 2: T-test, P=2, MLP layers=[100]              → 74.9%, IQR 31.8,  8.9–97.5  (ratio 2.2×)
  Row 3: PCA,    P=6, SVM kernel=rbf                → 59.9%, IQR  8.3, 44.1–74.3
  Row 4: PCA,    P=2, SVM kernel=rbf                → 55.0%, IQR 14.8, 24.7–78.6  (ratio 1.8×)

Tolerance: ±0.5 pp on median/IQR/min/max, ±0.05× on ratio.
"""

import pandas as pd, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
CSV = f"{REPO}/data/results/ad/ad_all_experiments_combined.csv"

# (label, exp, model, hp_substring, claimed_median%, claimed_iqr_pp, claimed_min%, claimed_max%, claimed_ratio_or_None)
ROWS = [
    ("T-test P=6 MLP[100]", "ANOVA_L_6_Random", "MLP",     '"hidden_layer_sizes": [100]',  72.6, 14.3, 54.9, 90.0, None),
    ("T-test P=2 MLP[100]", "ANOVA_L_2_Random", "MLP",     '"hidden_layer_sizes": [100]',  74.9, 31.8,  8.9, 97.5, 2.2),
    ("PCA P=6 SVM rbf",     "PCA_L_6_Random",   "SVM",     '"kernel": "rbf"',              59.9,  8.3, 44.1, 74.3, None),
    ("PCA P=2 SVM rbf",     "PCA_L_2_Random",   "SVM",     '"kernel": "rbf"',              55.0, 14.8, 24.7, 78.6, 1.8),
]
TOL = dict(median=0.5, iqr=0.5, minmax=0.5, ratio=0.05)


def main():
    df = pd.read_csv(CSV)
    diff_rows, table_rows = [], []
    print(f"{'Row':<24s} | {'med%':>7s} {'IQR':>6s} {'min':>6s} {'max':>6s} | status")
    print("-"*70)
    prev_iqr_p6 = {}
    for label, exp, model, hp_sub, p_med, p_iqr, p_min, p_max, p_ratio in ROWS:
        sub = df[(df["experiment"]==exp) & (df["model"]==model) & (df["hyperparams"].str.contains(hp_sub, regex=False, na=False))]
        accs = sub["test_accuracy"].values * 100
        med = np.median(accs)
        q1, q3 = np.percentile(accs, [25, 75])
        iqr = q3 - q1
        amin, amax = accs.min(), accs.max()
        ratio = None
        if p_ratio is not None:
            # ratio = current IQR / corresponding P=6 IQR (look up by feature pipeline)
            feat = "T-test" if "ANOVA" in exp else "PCA"
            if feat in prev_iqr_p6:
                ratio = iqr / prev_iqr_p6[feat]
        else:
            feat = "T-test" if "ANOVA" in exp else "PCA"
            prev_iqr_p6[feat] = iqr

        table_rows.append({"row":label,"n_folds":len(accs),"median_pct":round(med,2),"iqr_pp":round(iqr,2),
                           "min_pct":round(amin,2),"max_pct":round(amax,2),"ratio":round(ratio,2) if ratio else None})

        cells = [("median",p_med,med,TOL["median"]),
                 ("iqr",p_iqr,iqr,TOL["iqr"]),
                 ("min",p_min,amin,TOL["minmax"]),
                 ("max",p_max,amax,TOL["minmax"])]
        if p_ratio is not None and ratio is not None:
            cells.append(("ratio",p_ratio,ratio,TOL["ratio"]))
        for cell, claim, comp, tol in cells:
            d = comp - claim
            stat = "PASS" if abs(d) <= tol else "FAIL"
            diff_rows.append({"row":label,"cell":cell,"claimed":claim,"computed":round(comp,3),"delta":round(d,3),"status":stat,"tolerance":tol})

        ok = all(c["status"]=="PASS" for c in diff_rows[-len(cells):])
        print(f"{label:<24s} | {med:>6.1f} {iqr:>6.1f} {amin:>6.1f} {amax:>6.1f} | {'✓' if ok else '✗'}")

    diff = pd.DataFrame(diff_rows)
    out_diff = OUTDIR / "diff_T04.csv"
    out_table = OUTDIR / "table_T04_recomputed.csv"
    diff.to_csv(out_diff, index=False)
    pd.DataFrame(table_rows).to_csv(out_table, index=False)
    n_pass = (diff["status"]=="PASS").sum()
    print(f"\nT04 RESULT: {n_pass}/{len(diff)} cells PASS")
    if n_pass < len(diff):
        print(diff[diff["status"]!="PASS"].to_string(index=False))

if __name__ == "__main__":
    main()
