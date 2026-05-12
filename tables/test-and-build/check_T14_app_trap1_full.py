#!/usr/bin/env python3
"""
T14 Sanity Check — Table 14 (tab:app_trap1_full) — Appendix A.

Paper structure: per disease × {T-test, PCA} pipeline:
  | T-test P=6 (disj.) Min | Max | T-test infl pp Min | Max
  | PCA P=6 (disj.) Min | Max  | PCA infl pp Min | Max

Aggregation (canonical):
  For each (disease, pipeline, model): best HP by median test_accuracy → median across 50 folds
  Min/Max columns: range over the 4 model values

Inflation = subject-overlap median - LPSO P=6 median (per model), then min/max across 4 models.
W_C overlap uses test_accuracy (epoch) from W_C seed dirs, best HP per model, mean across 10 seeds.

Tolerance: ±0.01 on accuracies (paper rounds to 2 dp), ±2 pp on inflation Min/Max.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
ROOT = REPO
KNN_K7 = 7
MODELS = ["MLP", "XGBoost", "SVM", "KNN"]
MODEL_NORMALIZE = {"KNN":"KNN","SVM":"SVM","XGBoost":"XGBoost",
                   "MLP_(Neural_Network)":"MLP","MLP (Neural Network)":"MLP"}

# Paper claims (line 729-733 in .tex)
PAPER = {
    # disease: (T_disj_min, T_disj_max, T_infl_min, T_infl_max, P_disj_min, P_disj_max, P_infl_min, P_infl_max)
    "AD":     (0.70, 0.74,  +5, +26, 0.55, 0.60, +23, +41),
    "FTD":    (0.62, 0.65, +13, +36, 0.53, 0.54, +29, +45),
    "MDD EC": (0.79, 0.83,  +5, +12, 0.49, 0.63, +28, +45),
    "MDD EO": (0.75, 0.80, +10, +16, 0.50, 0.64, +25, +45),
    "SCZ":    (0.56, 0.60, +19, +32, 0.45, 0.56, +30, +41),
}
TOL = dict(acc=0.01, infl=2)

# Data sources
COMB_CSV = {
    "AD":     ROOT/"data/results/ad/ad_all_experiments_combined.csv",
    "FTD":    ROOT/"data/results/ftd/ftd_all_experiments_combined.csv",
    "MDD EC": ROOT/"data/results/mdd/mdd_all_experiments_combined.csv",
    "MDD EO": ROOT/"data/results/mdd/mdd_all_experiments_combined.csv",
    "SCZ":    ROOT/"data/results/scz/scz_all_experiments_combined.csv",
}
EXP_NAMES = {
    "AD":     dict(ANOVA="ANOVA_L_6_Random", PCA="PCA_L_6_Random"),
    "FTD":    dict(ANOVA="ANOVA_L_6_FTD",    PCA="PCA_L_6_FTD"),
    "MDD EC": dict(ANOVA="ANOVA_L_6_EC",     PCA="PCA_L_6_EC"),
    "MDD EO": dict(ANOVA="ANOVA_L_6_EO",     PCA="PCA_L_6_EO"),
    "SCZ":    dict(ANOVA="ANOVA_L_6_SCZ",    PCA="PCA_L_6_SCZ"),
}
WC_PREFIX = {
    "AD":     ("{pipe}_W_C_ad_cntrl_seed{seed}", ""),
    "FTD":    ("{pipe}_W_C_ftd_vs_cntrl_seed{seed}", "data/ftd_vs_C/intra-subject_seed_runs"),
    "MDD EC": ("{pipe}_W_C_mdd_cs_cntrl_seed{seed}_EC", "data/mdd_vs_cntrl"),
    "MDD EO": ("{pipe}_W_C_mdd_cs_cntrl_seed{seed}_eyesopen_EO", "data/mdd_vs_cntrl"),
    "SCZ":    ("{pipe}_W_C_scz_cntrl_seed{seed}", "scz_vs_cntrl_all_results"),
}


def disjoint_per_model(comb_csv, exp_name):
    df = pd.read_csv(comb_csv)
    sub = df[df["experiment"] == exp_name]
    out = {}
    for m in MODELS:
        mm = sub[sub["model"] == m]
        if mm.empty: continue
        if m == "KNN":
            mk = mm[mm["hyperparams"].str.contains(f'"n_neighbors": {KNN_K7}', na=False)]
            if mk.empty: continue
            out[m] = mk["test_accuracy"].median()
        else:
            best = mm.groupby("hyperparams")["test_accuracy"].median().idxmax()
            out[m] = mm[mm["hyperparams"] == best]["test_accuracy"].median()
    return out


_WC_CACHE = None
def _wc_df():
    global _WC_CACHE
    if _WC_CACHE is None:
        _WC_CACHE = pd.read_csv(REPO / "data/w_c_per_model_accuracies.csv")
    return _WC_CACHE

_COHORT_KEY = {"AD": "ad", "FTD": "ftd", "MDD EC": "mdd_ec", "MDD EO": "mdd_eo", "SCZ": "scz"}

def overlap_per_model(cohort_name, pipe_prefix):
    df = _wc_df()
    sub = df[(df["cohort"] == _COHORT_KEY[cohort_name]) & (df["pipeline"] == pipe_prefix)]
    out = {}
    for m in MODELS:
        accs = sub[sub["model"] == m]["accuracy"].values
        if len(accs):
            out[m] = float(accs.mean())
    return out


def main():
    diff_rows, table_rows = [], []
    print(f"{'D':6s} | {'T-disj':>13s} {'T-infl pp':>13s} {'P-disj':>13s} {'P-infl pp':>13s}")
    print("-" * 80)
    for name in PAPER:
        # T-test
        dj_t = disjoint_per_model(COMB_CSV[name], EXP_NAMES[name]["ANOVA"])
        ov_t = overlap_per_model(name, "ANOVA")
        # PCA
        dj_p = disjoint_per_model(COMB_CSV[name], EXP_NAMES[name]["PCA"])
        ov_p = overlap_per_model(name, "PCA")

        t_disj_min, t_disj_max = min(dj_t.values()), max(dj_t.values())
        p_disj_min, p_disj_max = min(dj_p.values()), max(dj_p.values())

        t_infls = [(ov_t[m] - dj_t[m]) * 100 for m in MODELS if m in dj_t and m in ov_t]
        p_infls = [(ov_p[m] - dj_p[m]) * 100 for m in MODELS if m in dj_p and m in ov_p]
        t_infl_min, t_infl_max = min(t_infls), max(t_infls)
        p_infl_min, p_infl_max = min(p_infls), max(p_infls)

        p = PAPER[name]
        table_rows.append({"disease":name,
                           "T_disj_min":round(t_disj_min,3),"T_disj_max":round(t_disj_max,3),
                           "T_infl_min_pp":round(t_infl_min,1),"T_infl_max_pp":round(t_infl_max,1),
                           "P_disj_min":round(p_disj_min,3),"P_disj_max":round(p_disj_max,3),
                           "P_infl_min_pp":round(p_infl_min,1),"P_infl_max_pp":round(p_infl_max,1)})
        cells = [
            ("T_disj_min", p[0], t_disj_min, TOL["acc"]),
            ("T_disj_max", p[1], t_disj_max, TOL["acc"]),
            ("T_infl_min", p[2], t_infl_min, TOL["infl"]),
            ("T_infl_max", p[3], t_infl_max, TOL["infl"]),
            ("P_disj_min", p[4], p_disj_min, TOL["acc"]),
            ("P_disj_max", p[5], p_disj_max, TOL["acc"]),
            ("P_infl_min", p[6], p_infl_min, TOL["infl"]),
            ("P_infl_max", p[7], p_infl_max, TOL["infl"]),
        ]
        for cell, claim, comp, tol in cells:
            d = comp - claim
            stat = "PASS" if abs(d) <= tol else "FAIL"
            diff_rows.append({"disease":name,"cell":cell,"claimed":claim,"computed":round(comp,3),"delta":round(d,3),"status":stat,"tolerance":tol})
        print(f"{name:6s} | {t_disj_min:.2f}-{t_disj_max:.2f}/{p[0]:.2f}-{p[1]:.2f} {t_infl_min:+.0f}/{t_infl_max:+.0f}/{p[2]:+d}/{p[3]:+d} {p_disj_min:.2f}-{p_disj_max:.2f}/{p[4]:.2f}-{p[5]:.2f} {p_infl_min:+.0f}/{p_infl_max:+.0f}/{p[6]:+d}/{p[7]:+d}")

    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUTDIR/"diff_T14.csv", index=False)
    pd.DataFrame(table_rows).to_csv(OUTDIR/"table_T14_recomputed.csv", index=False)
    n_pass = (diff["status"]=="PASS").sum()
    print(f"\nT14 RESULT: {n_pass}/{len(diff)} cells PASS")
    if n_pass < len(diff):
        print(diff[diff["status"]!="PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
