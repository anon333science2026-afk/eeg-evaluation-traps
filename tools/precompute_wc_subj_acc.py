#!/usr/bin/env python3
"""Pre-compute per-(cohort, pipeline, seed, model, task) Sub. Acc. (majority-vote)
from W_C test_predictions parquets. Writes data/w_c_subject_acc.csv.

Mirrors build_subj_acc_tables.py overlap_subj() exactly:
  - first 3 tasks per (seed, model) in sorted order
  - per-subject majority vote over epoch predictions
  - row per (cohort, pipeline, seed, model, task_idx, subject_acc)
"""
import os
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/w_c_subject_acc.csv"
# Override SRC env var to point at your local raw pipeline-output tree.
SRC = Path(os.environ.get("SRC", "/path/to/eeg-pipeline-outputs"))

COHORTS = {
    "ad":     (SRC,                                                              "{pipe}_W_C_ad_cntrl_seed{seed}"),
    "ftd":    (SRC / "data/ftd_vs_C/intra-subject_seed_runs",                    "{pipe}_W_C_ftd_vs_cntrl_seed{seed}"),
    "mdd_ec": (SRC / "data/mdd_vs_cntrl",                                        "{pipe}_W_C_mdd_cs_cntrl_seed{seed}_EC"),
    "mdd_eo": (SRC / "data/mdd_vs_cntrl",                                        "{pipe}_W_C_mdd_cs_cntrl_seed{seed}_eyesopen_EO"),
    "scz":    (SRC / "scz_vs_cntrl_all_results",                                 "{pipe}_W_C_scz_cntrl_seed{seed}"),
}


def main():
    rows = []
    for cohort, (base, tpl) in COHORTS.items():
        for pipe in ("ANOVA", "PCA"):
            for seed in range(42, 52):
                seed_dir = base / tpl.format(pipe=pipe, seed=seed) / "ml_results_grid_search"
                if not seed_dir.exists():
                    continue
                for model_dir in sorted(seed_dir.iterdir()):
                    if not model_dir.is_dir() or model_dir.name in ["graphs", "debug"]:
                        continue
                    wss = model_dir / "within_subject_split"
                    if not wss.exists():
                        continue
                    tasks = sorted([t for t in wss.iterdir()
                                    if t.is_dir() and (t / "test_predictions.parquet").exists()])[:3]
                    for ti, task_dir in enumerate(tasks):
                        pred = pd.read_parquet(task_dir / "test_predictions.parquet")
                        per_sub = pred.groupby("SubjectID").apply(
                            lambda g: ((g["prediction"].mode().iloc[0]
                                        if not g["prediction"].mode().empty
                                        else g["prediction"].iloc[0]) == g["label"].iloc[0]),
                            include_groups=False)
                        rows.append({
                            "cohort": cohort, "pipeline": pipe, "seed": seed,
                            "model": model_dir.name, "task_idx": ti,
                            "subject_acc": float(per_sub.mean()),
                        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows → {OUT}")
    print(df.groupby(["cohort", "pipeline"]).size().to_string())


if __name__ == "__main__":
    main()
