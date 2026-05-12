#!/usr/bin/env python3
"""Walk every ANOVA_W_C / PCA_W_C seed dir across cohorts and emit
processed_data/w_c_per_model_accuracies.csv with columns:
  cohort, pipeline, seed, model, accuracy

Replicates the same per-model logic used by T05 overlap_per_model() and
T09 overlap_t_test_stats() so downstream scripts can read this CSV
instead of the ~930 MB of raw ml_results_grid_search dirs.

For each (cohort, pipe, seed):
- KNN: read hyperparameter_comparison.csv and take row with n_neighbors=7
- Others: walk rglob("results.json"), take MAX accuracy across HP search
"""
import json
import os
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "processed_data/w_c_per_model_accuracies.csv"
# Override the SRC env var to point at your local raw pipeline-output tree.
#   SRC=/your/path/to/eeg-pipeline-outputs python tools/precompute_wc_summary.py
SRC = Path(os.environ.get("SRC", "/path/to/eeg-pipeline-outputs"))
KNN_K = 7
MODELS = {"MLP", "XGBoost", "SVM", "KNN"}
NORMALIZE = {"XGB": "XGBoost", "xgboost": "XGBoost", "xgb": "XGBoost",
             "mlp": "MLP", "svm": "SVM", "knn": "KNN", "Knn": "KNN",
             "MLP_(Neural_Network)": "MLP"}

# Per-cohort: base_dir, prefix_template
COHORTS = {
    "ad":     (SRC,                                                              "{pipe}_W_C_ad_cntrl_seed{seed}"),
    "ftd":    (SRC / "data/ftd_vs_C/intra-subject_seed_runs",                    "{pipe}_W_C_ftd_vs_cntrl_seed{seed}"),
    "mdd_ec": (SRC / "data/mdd_vs_cntrl",                                        "{pipe}_W_C_mdd_cs_cntrl_seed{seed}_EC"),
    "mdd_eo": (SRC / "data/mdd_vs_cntrl",                                        "{pipe}_W_C_mdd_cs_cntrl_seed{seed}_eyesopen_EO"),
    "scz":    (SRC / "scz_vs_cntrl_all_results",                                 "{pipe}_W_C_scz_cntrl_seed{seed}"),
}


def per_model_for_seed(seed_dir: Path):
    out = {}
    if not seed_dir.exists():
        return out
    for md in seed_dir.iterdir():
        if not md.is_dir():
            continue
        model = NORMALIZE.get(md.name, md.name)
        if model not in MODELS:
            continue
        if model == "KNN":
            hp_csv = md / "hyperparameter_comparison.csv"
            if hp_csv.exists():
                df = pd.read_csv(hp_csv)
                k_col = next((c for c in df.columns if "neighbor" in c.lower()), None)
                acc_col = next((c for c in df.columns if "accuracy" in c.lower()), None)
                if k_col and acc_col:
                    row = df[df[k_col].astype(str) == str(KNN_K)]
                    if not row.empty:
                        out["KNN"] = float(row[acc_col].iloc[0])
            continue
        best = None
        for rj in md.rglob("results.json"):
            try:
                d = json.loads(rj.read_text())
                acc = (d.get("test_results", {}).get("accuracy")
                       or d.get("test_accuracy")
                       or d.get("detailed_results", {}).get("test_accuracy"))
                if acc is not None:
                    acc = float(acc)
                    if best is None or acc > best:
                        best = acc
            except Exception:
                pass
        if best is not None:
            out[model] = best
    return out


def main():
    rows = []
    for cohort, (base, tpl) in COHORTS.items():
        for pipe in ("ANOVA", "PCA"):
            for seed in range(42, 52):
                exp = tpl.format(pipe=pipe, seed=seed)
                seed_dir = base / exp / "ml_results_grid_search"
                accs = per_model_for_seed(seed_dir)
                for model, acc in accs.items():
                    rows.append({"cohort": cohort, "pipeline": pipe, "seed": seed,
                                 "model": model, "accuracy": acc})
                if not accs:
                    print(f"  ! missing: {seed_dir}")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} rows → {OUT}")
    print(df.groupby(["cohort", "pipeline"]).size().to_string())


if __name__ == "__main__":
    main()
