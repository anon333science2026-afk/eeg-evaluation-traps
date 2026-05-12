#!/usr/bin/env python3
"""
T16 Sanity Check — Table 16 (tab:app_trap3_full) — Appendix C.

Paper structure: per disease × pipeline × P:
  | Ep. Acc. Min/Max | Sub. Acc. Min/Max | Gap pp Min/Max |

Range = min/max over the 4 model values per (disease, pipeline, P).
Gap = subject - epoch (positive = subject > epoch). Paper convention is
"Gap = epoch - subject" so paper's negative means subject > epoch.

────────────────────────────────────────────────────────────────────────────
KNOWN DIVERGENCE — to be fixed for camera-ready
────────────────────────────────────────────────────────────────────────────
T16 in the submitted paper was built BEFORE the Round-2 canonical
aggregation rule was finalized. The rest of the paper (T01–T15, all
figures) uses the canonical rule:
    median across folds, KNN locked to k=7.

T16, however, was generated with the legacy rule:
    Sub. Acc. = MEAN across folds, KNN best-HP picked by median (NOT locked to k=7).
    Ep. Acc.  = MEDIAN across folds, KNN best-HP picked by median.

This script runs BOTH rules:
  - `--rule canonical` (default): canonical median rule. Used to flag T16 as
    NOT matching the rest of the paper's aggregation. Expected to FAIL many
    cells. This is the audit result we accept for the submission as a known
    divergence.
  - `--rule legacy`: pre-Round-2 mean rule. Used to reproduce the paper's
    Table 16 exactly. Expected to PASS.

For camera-ready: switch the default to canonical, regenerate Table 16 from
the canonical-recomputed numbers (saved to `table_T16_canonical.csv`), and
update the LaTeX values in the paper. The legacy rule code stays here as
provenance so reviewers can verify the submitted-version numbers.

Quantization note: Sub. Acc. per fold is a fraction k/P (k correct out of P
held-out subjects). Median across folds collapses to one of those discrete
values (0.500, 0.667, 0.833, 1.000 for P=6). Mean does not. Under the
canonical rule, all 4 models often share the same median Sub. Acc., so
sub_min == sub_max — this is real, not a bug.

Tolerance: ±0.005 on accuracies, ±1 pp on gap.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
ROOT = REPO
KNN_K7 = '"n_neighbors": 7'
MODELS = ["MLP","XGBoost","SVM","KNN"]

# Paper claims (lines 769-792). Format: (ep_min, ep_max, subj_min, subj_max, gap_min_pp, gap_max_pp)
# Paper gap = epoch - subject (negative means subject > epoch)
PAPER = {
    ("AD",     "T-test", 6): (0.699, 0.727, 0.793, 0.857, -14.1, -6.6),
    ("AD",     "T-test", 2): (0.660, 0.682, 0.720, 0.770, -11.0, -5.4),
    ("AD",     "PCA",    6): (0.549, 0.596, 0.550, 0.623, -5.1,  +3.6),
    ("AD",     "PCA",    2): (0.533, 0.560, 0.490, 0.600, -6.1,  +4.3),
    ("FTD",    "T-test", 6): (0.605, 0.631, 0.623, 0.650, -4.2,  -0.5),
    ("FTD",    "T-test", 2): (0.604, 0.624, 0.580, 0.620, +0.4,  +4.1),
    ("FTD",    "PCA",    6): (0.542, 0.555, 0.500, 0.520, +2.2,  +4.5),
    ("FTD",    "PCA",    2): (0.556, 0.558, 0.500, 0.500, +5.6,  +5.8),
    ("MDD EC", "T-test", 6): (0.796, 0.824, 0.830, 0.883, -5.9,  -3.4),
    ("MDD EC", "T-test", 2): (0.792, 0.823, 0.800, 0.870, -5.1,  -0.8),
    ("MDD EC", "PCA",    6): (0.487, 0.642, 0.500, 0.653, -1.3,  -0.3),
    ("MDD EC", "PCA",    2): (0.484, 0.666, 0.500, 0.710, -4.4,  +0.2),
    ("MDD EO", "T-test", 6): (0.748, 0.794, 0.777, 0.847, -5.4,  -2.8),
    ("MDD EO", "T-test", 2): (0.768, 0.867, 0.760, 0.940, -8.5,  +0.8),
    ("MDD EO", "PCA",    6): (0.494, 0.630, 0.500, 0.653, -2.3,  -0.1),
    ("MDD EO", "PCA",    2): (0.500, 0.700, 0.500, 0.740, -4.0,  +0.3),
    ("SCZ",    "T-test", 6): (0.556, 0.597, 0.587, 0.690, -10.1, -2.9),
    ("SCZ",    "T-test", 2): (0.520, 0.566, 0.440, 0.650, -8.4,  +8.0),
    ("SCZ",    "PCA",    6): (0.473, 0.557, 0.497, 0.567, -2.7,  +0.4),
    ("SCZ",    "PCA",    2): (0.464, 0.537, 0.500, 0.500, -3.6,  +3.7),
}
TOL = dict(acc=0.005, gap=1.0)

# Data sources
SOURCE = {
    "AD":     dict(comb=ROOT/"processed_data/results/ad/ad_all_experiments_combined.csv",
                   fold=ROOT/"processed_data/results/ad/ad_fold_epoch_vs_subject.csv",
                   pcol="pipeline", cond=None,
                   exp={"T-test":{6:"ANOVA_L_6_Random",2:"ANOVA_L_2_Random"},
                        "PCA":   {6:"PCA_L_6_Random",  2:"PCA_L_2_Random"}}),
    "FTD":    dict(comb=ROOT/"processed_data/results/ftd/ftd_all_experiments_combined.csv",
                   fold=ROOT/"processed_data/results/ftd/ftd_fold_epoch_vs_subject.csv",
                   pcol="pipeline", cond=None,
                   exp={"T-test":{6:"ANOVA_L_6_FTD",2:"ANOVA_L_2_FTD"},
                        "PCA":   {6:"PCA_L_6_FTD",  2:"PCA_L_2_FTD"}}),
    "MDD EC": dict(comb=ROOT/"processed_data/results/mdd/mdd_all_experiments_combined.csv",
                   fold=ROOT/"processed_data/results/mdd/mdd_fold_epoch_vs_subject.csv",
                   pcol="feature_set", cond="EC",
                   exp={"T-test":{6:"ANOVA_L_6_EC",2:"ANOVA_L_2_EC"},
                        "PCA":   {6:"PCA_L_6_EC",  2:"PCA_L_2_EC"}}),
    "MDD EO": dict(comb=ROOT/"processed_data/results/mdd/mdd_all_experiments_combined.csv",
                   fold=ROOT/"processed_data/results/mdd/mdd_fold_epoch_vs_subject.csv",
                   pcol="feature_set", cond="EO",
                   exp={"T-test":{6:"ANOVA_L_6_EO",2:"ANOVA_L_2_EO"},
                        "PCA":   {6:"PCA_L_6_EO",  2:"PCA_L_2_EO"}}),
    "SCZ":    dict(comb=ROOT/"processed_data/results/scz/scz_all_experiments_combined.csv",
                   fold=ROOT/"processed_data/results/scz/scz_fold_epoch_vs_subject.csv",
                   pcol="pipeline", cond=None,
                   exp={"T-test":{6:"ANOVA_L_6_SCZ",2:"ANOVA_L_2_SCZ"},
                        "PCA":   {6:"PCA_L_6_SCZ",  2:"PCA_L_2_SCZ"}}),
}
# Pipeline-name → value in the fold CSV's pipeline/feature_set column.
# AD/FTD/SCZ use a "pipeline" column with values FTest/PCA.
# MDD uses a "feature_set" column with values ANOVA/PCA — same concept, different vocabulary.
PIPE_FOLD_MAP = {
    "pipeline":    {"T-test": "FTest", "PCA": "PCA"},
    "feature_set": {"T-test": "ANOVA", "PCA": "PCA"},
}


def per_model_stats(disease, pipeline, P, rule="canonical"):
    """Return {model: (ep, sub)} for one (disease, pipeline, P) cell.

    rule="canonical" — median across folds, KNN locked to k=7.
    rule="legacy"    — Sub. Acc. is MEAN across folds, KNN best-HP picked by
                       median test_accuracy from the combined CSV (NOT locked).
                       Ep. Acc. stays median (per paper-Table-16 convention).
    """
    s = SOURCE[disease]
    comb = pd.read_csv(s["comb"])
    fold = pd.read_csv(s["fold"])
    fold = fold[fold[s["pcol"]] == PIPE_FOLD_MAP[s["pcol"]][pipeline]]
    if s["cond"] and "condition" in fold.columns:
        fold = fold[fold["condition"] == s["cond"]]
    fold = fold[fold["P"] == P]
    exp_name = s["exp"][pipeline][P]
    out = {}
    for m in MODELS:
        if m == "KNN" and rule == "canonical":
            best_hp = '{"metric": "euclidean", "n_neighbors": 7, "weights": "uniform"}'
        else:
            mm = comb[(comb["experiment"] == exp_name) & (comb["model"] == m)]
            if mm.empty:
                continue
            best_hp = mm.groupby("hyperparams")["test_accuracy"].median().idxmax()
        sub = fold[(fold["model"] == m) & (fold["hyperparams"] == best_hp)]
        if sub.empty:
            continue
        ep_stat = sub["epoch_acc"].median()
        sub_stat = sub["subject_acc"].mean() if rule == "legacy" else sub["subject_acc"].median()
        out[m] = (ep_stat, sub_stat)
    return out


def run_rule(rule):
    """Recompute the full table under one aggregation rule and return (diff_df, table_df)."""
    diff_rows, table_rows = [], []
    print(f"\n=== Rule: {rule.upper()} ===")
    print(f"{'D':6s} {'Pipe':>6s} {'P':>2s} | {'ep min/max':>14s} {'sub min/max':>14s} {'gap min/max':>14s}")
    print("-" * 90)
    for (name, pipe, P), p in PAPER.items():
        per_model = per_model_stats(name, pipe, P, rule=rule)
        if not per_model:
            print(f"{name:6s} {pipe:>6s} {P:>2d} | (no data)")
            continue
        eps = [v[0] for v in per_model.values()]
        subs = [v[1] for v in per_model.values()]
        gaps = [(per_model[m][0] - per_model[m][1]) * 100 for m in per_model]
        ep_min, ep_max = min(eps), max(eps)
        sub_min, sub_max = min(subs), max(subs)
        gap_min, gap_max = min(gaps), max(gaps)

        table_rows.append({"disease": name, "pipeline": pipe, "P": P,
                           "ep_min": round(ep_min, 3), "ep_max": round(ep_max, 3),
                           "sub_min": round(sub_min, 3), "sub_max": round(sub_max, 3),
                           "gap_min_pp": round(gap_min, 1), "gap_max_pp": round(gap_max, 1)})
        cells = [
            ("ep_min",  p[0], ep_min,  TOL["acc"]),
            ("ep_max",  p[1], ep_max,  TOL["acc"]),
            ("sub_min", p[2], sub_min, TOL["acc"]),
            ("sub_max", p[3], sub_max, TOL["acc"]),
            ("gap_min", p[4], gap_min, TOL["gap"]),
            ("gap_max", p[5], gap_max, TOL["gap"]),
        ]
        for cell, claim, comp, tol in cells:
            d = comp - claim
            stat = "PASS" if abs(d) <= tol else "FAIL"
            diff_rows.append({"disease": name, "pipeline": pipe, "P": P, "cell": cell,
                              "claimed": claim, "computed": round(comp, 3), "delta": round(d, 3),
                              "status": stat, "tolerance": tol})
        print(f"{name:6s} {pipe:>6s} {P:>2d} | {ep_min:.3f}-{ep_max:.3f} ({p[0]:.3f}-{p[1]:.3f}) {sub_min:.3f}-{sub_max:.3f} ({p[2]:.3f}-{p[3]:.3f}) {gap_min:+.1f}/{gap_max:+.1f} ({p[4]:+.1f}/{p[5]:+.1f})")
    return pd.DataFrame(diff_rows), pd.DataFrame(table_rows)


def main():
    rule = "canonical"
    if "--rule" in sys.argv:
        rule = sys.argv[sys.argv.index("--rule") + 1]
    if rule not in ("canonical", "legacy", "both"):
        print(f"Unknown rule: {rule}. Use canonical | legacy | both")
        sys.exit(2)

    rules = ["canonical", "legacy"] if rule == "both" else [rule]
    summaries = {}
    for r in rules:
        diff, tbl = run_rule(r)
        suffix = "_legacy" if r == "legacy" else "_canonical"
        diff.to_csv(OUTDIR / f"diff_T16{suffix}.csv", index=False)
        tbl.to_csv(OUTDIR / f"table_T16{suffix}.csv", index=False)
        n_pass = (diff["status"] == "PASS").sum()
        summaries[r] = (n_pass, len(diff))
        print(f"\nT16 RESULT ({r}): {n_pass}/{len(diff)} cells PASS")
        if n_pass < len(diff):
            print(f"  Failures saved to diff_T16{suffix}.csv")

    # Also keep legacy filenames for backwards-compat with older audit pipelines
    # that expect diff_T16.csv / table_T16_recomputed.csv.
    primary = "legacy" if "legacy" in summaries else "canonical"
    pd.read_csv(OUTDIR / f"diff_T16_{primary}.csv").to_csv(OUTDIR / "diff_T16.csv", index=False)
    pd.read_csv(OUTDIR / f"table_T16_{primary}.csv").to_csv(OUTDIR / "table_T16_recomputed.csv", index=False)

    print("""
────────────────────────────────────────────────────────────────────────────
T16 is a KNOWN DIVERGENCE in the submitted paper.

The paper's Table 16 was generated with the LEGACY aggregation rule
(Sub. Acc. = mean across folds, KNN best-HP unlocked). The rest of the
paper (T01-T15, all figures) uses the CANONICAL rule (median across folds,
KNN locked to k=7) that was finalized in Round-2 of the audit.

  - Run with `--rule legacy`    → reproduces the submitted Table 16 (expected PASS).
  - Run with `--rule canonical` → applies the post-Round-2 rule used by the rest
                                  of the paper (expected FAIL — this is the
                                  whole reason T16 needs to be updated).
  - Run with `--rule both`      → produces both side-by-side.

TO DO FOR CAMERA-READY:
  1. Regenerate Table 16 in the paper from `table_T16_canonical.csv`.
  2. Update the LaTeX claim values to the canonical numbers.
  3. Re-run this script with `--rule canonical` and confirm 96/96 PASS.
  4. Add a short footnote in the appendix noting the aggregation rule
     change (consistency with the canonical rule used elsewhere).
────────────────────────────────────────────────────────────────────────────""")


if __name__ == "__main__":
    main()
