#!/usr/bin/env python3
"""
T5 Sanity Check — Table 5 (tab:replication_trap1) — Trap 1 across diseases.

Paper post-restructure (line 296):
  | Dataset | Ep. Acc. P=6 (disj.) | Ep. Acc. overlap | Inflation (pp) |

Aggregation rule (canonical, locked Round 2):
  best HP per model (by median test_accuracy) → MEDIAN across 50 folds → mean of 4 models

Disjoint side: <dis>_all_experiments_combined.csv, T-test (ANOVA), P=6, LPSO Random-50.
Overlap side: <prefix>_seed{42..51}/ml_results_grid_search/<model>/, KNN locked to k=7.

Tolerance: ±0.005 on Ep. Acc., ±0.5 pp on inflation.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
KNN_K7 = 7
MODELS = ["MLP", "XGBoost", "SVM", "KNN"]
ROOT = REPO

DJ_CONFIG = [
    # (name, combined_csv, exp_name, feat_set_value)
    ("AD",     "processed_data/results/ad/ad_all_experiments_combined.csv",         "ANOVA_L_6_Random",  "ANOVA"),
    ("FTD",    "processed_data/results/ftd/ftd_all_experiments_combined.csv",            "ANOVA_L_6_FTD",     "ANOVA"),
    ("MDD EC", "processed_data/results/mdd/mdd_all_experiments_combined.csv",        "ANOVA_L_6_EC",      "ANOVA"),
    ("MDD EO", "processed_data/results/mdd/mdd_all_experiments_combined.csv",        "ANOVA_L_6_EO",      "ANOVA"),
    ("SCZ",    "processed_data/results/scz/scz_all_experiments_combined.csv", "ANOVA_L_6_SCZ",     "ANOVA"),
]
WC_CONFIG = {
    "AD":     ("ANOVA_W_C_ad_cntrl_seed{seed}", ""),
    "FTD":    ("ANOVA_W_C_ftd_vs_cntrl_seed{seed}", "data/ftd_vs_C/intra-subject_seed_runs"),
    "MDD EC": ("ANOVA_W_C_mdd_cs_cntrl_seed{seed}_EC", "data/mdd_vs_cntrl"),
    "MDD EO": ("ANOVA_W_C_mdd_cs_cntrl_seed{seed}_eyesopen_EO", "data/mdd_vs_cntrl"),
    "SCZ":    ("ANOVA_W_C_scz_cntrl_seed{seed}", "scz_vs_cntrl_all_results"),
}
PAPER = {  # (Ep.Acc. P=6 disj, Ep.Acc. overlap, Inflation pp)
    "AD":     (0.715, 0.851, 13.6),
    "FTD":    (0.627, 0.871, 24.3),
    "MDD EC": (0.814, 0.914, 10.0),
    "MDD EO": (0.781, 0.911, 13.0),
    "SCZ":    (0.573, 0.851, 27.8),
}
TOL = dict(disj=0.005, ov=0.005, infl=0.5)
MODEL_NORMALIZE = {"KNN":"KNN","SVM":"SVM","XGBoost":"XGBoost",
                   "MLP_(Neural_Network)":"MLP","MLP (Neural Network)":"MLP"}


def disjoint_per_model_median(comb_csv, exp_name):
    df = pd.read_csv(comb_csv)
    sub = df[df["experiment"] == exp_name]
    out = {}
    for model in MODELS:
        m = sub[sub["model"] == model]
        if m.empty: continue
        if model == "KNN":
            knn = m[m["hyperparams"].str.contains(f'"n_neighbors": {KNN_K7}', na=False)]
            if knn.empty: continue
            out[model] = knn["test_accuracy"].median()
        else:
            best_hp = m.groupby("hyperparams")["test_accuracy"].median().idxmax()
            out[model] = m[m["hyperparams"] == best_hp]["test_accuracy"].median()
    return out


_WC_CACHE = None
def _wc_df():
    global _WC_CACHE
    if _WC_CACHE is None:
        _WC_CACHE = pd.read_csv(REPO / "processed_data/w_c_per_model_accuracies.csv")
    return _WC_CACHE

# cohort name (matches T05 DJ_CONFIG first column) → key in w_c_per_model_accuracies.csv
_COHORT_KEY = {"AD": "ad", "FTD": "ftd", "MDD EC": "mdd_ec", "MDD EO": "mdd_eo", "SCZ": "scz"}

def overlap_per_model(cohort_name, pipeline="ANOVA"):
    df = _wc_df()
    key = _COHORT_KEY[cohort_name]
    sub = df[(df["cohort"] == key) & (df["pipeline"] == pipeline)]
    out = {}
    for model in MODELS:
        accs = sub[sub["model"] == model]["accuracy"].values
        if len(accs):
            out[model] = float(accs.mean())
    return out


def main():
    diff_rows = []
    table_rows = []
    print(f"{'D':6s} | {'Paper (disj/ov/infl)':<22s} | {'Computed':<22s} | status")
    print("-" * 80)
    for name, ccsv, exp, _ in DJ_CONFIG:
        dj = disjoint_per_model_median(ROOT / ccsv, exp)
        ov = overlap_per_model(name, pipeline="ANOVA")
        if not dj or not ov:
            print(f"{name}: missing data"); continue
        avg_dj = float(np.mean(list(dj.values())))
        avg_ov = float(np.mean(list(ov.values())))
        infl = (avg_ov - avg_dj) * 100
        p_dj, p_ov, p_infl = PAPER[name]
        table_rows.append({"dataset":name,"Ep_Acc_P6_disj":round(avg_dj,4),"Ep_Acc_overlap":round(avg_ov,4),"Inflation_pp":round(infl,2)})
        for cell, claim, comp, tol in [("Ep_Acc_P6_disj",p_dj,avg_dj,TOL["disj"]),
                                         ("Ep_Acc_overlap",p_ov,avg_ov,TOL["ov"]),
                                         ("Inflation_pp",p_infl,infl,TOL["infl"])]:
            d = comp - claim
            stat = "PASS" if abs(d) <= tol else "FAIL"
            diff_rows.append({"dataset":name,"cell":cell,"claimed":claim,"computed":round(comp,4),"delta":round(d,4),"status":stat,"tolerance":tol})
        ok = all(d["status"]=="PASS" for d in diff_rows[-3:])
        print(f"{name:6s} | {f'{p_dj:.3f} / {p_ov:.3f} / {p_infl:+.1f}':<22s} | {f'{avg_dj:.3f} / {avg_ov:.3f} / {infl:+.1f}':<22s} | {'✓' if ok else '✗'}")

    diff = pd.DataFrame(diff_rows)
    out_diff = OUTDIR / "diff_T05.csv"
    out_table = OUTDIR / "table_T05_recomputed.csv"
    diff.to_csv(out_diff, index=False)
    pd.DataFrame(table_rows).to_csv(out_table, index=False)
    n_pass = (diff["status"]=="PASS").sum()
    print(f"\nT05 RESULT: {n_pass}/{len(diff)} cells PASS")
    if n_pass < len(diff):
        print("\nFailures:")
        print(diff[diff["status"]!="PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
