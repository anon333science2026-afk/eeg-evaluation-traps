#!/usr/bin/env python3
"""
FTD vs C — Trap 3: Epoch vs Subject accuracy scatter.
Random-50 only: P=6 (circles) and P=2 (triangles).
Horizontal 1x2: PCA (left), T-test (right).

Dataset: FTD vs Control
  FTD: 23 subjects / Control: 29 subjects / Total: 52
  Epoch-level chance: 0.5422 (majority=cntrl)
  Subject-level chance: 29/52 = 0.5577

Adapted for NeurIPS figures output directory.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Helvetica"

IN_CSV  = Path(f"{REPO}/processed_data/results/ftd/ftd_fold_epoch_vs_subject.csv")
OUT_DIR = Path(f"{REPO}/figures/out")

CHANCE_EPOCH = 0.5965  # epoch-level (control majority; verified vs canonical parquet 2026-05-07)
CHANCE_SUBJ  = 29 / 52   # 0.5577

MODEL_ORDER  = ["MLP", "KNN", "SVM", "XGBoost"]
MODEL_COLORS = {
    "MLP":     "#1f77b4",
    "XGBoost": "#ff7f0e",
    "SVM":     "#2ca02c",
    "KNN":     "#d62728",
}
P_MARKERS = {6: "o", 2: "^"}


def display_name(pipeline):
    return "T-test" if pipeline == "FTest" else pipeline


# Load and aggregate
merged = pd.read_csv(IN_CSV)
print(f"Loaded: {len(merged):,} rows")

merged = merged[merged["strategy"] == "Random-50"].copy()
print(f"After Random-50 filter: {len(merged):,} rows")
print(f"  P values: {sorted(merged['P'].unique())}")
print(f"  Pipelines: {merged['pipeline'].unique()}")

agg_rows = []
for (pipeline, P, model, hyperparams), grp in merged.groupby(
        ["pipeline", "P", "model", "hyperparams"]):
    mean_epoch = grp["epoch_acc"].mean()
    total_subjects = grp["n_subjects_fold"].sum()
    subjects_correct = (grp["subject_acc"] * grp["n_subjects_fold"]).sum()
    frac_correct = subjects_correct / total_subjects if total_subjects > 0 else 0.0
    agg_rows.append({
        "pipeline": pipeline,
        "P": P,
        "model": model,
        "mean_epoch": mean_epoch,
        "frac_subjects_correct": frac_correct,
    })
agg_df = pd.DataFrame(agg_rows)
print(f"Aggregated: {len(agg_df)} points")


# Figure: 1x2 — PCA left (col 0), T-test right (col 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), gridspec_kw={"wspace": 0.32})

added_labels = set()
for ai, pipeline in enumerate(["PCA", "FTest"]):
    ax = axes[ai]
    panel = agg_df[agg_df["pipeline"] == pipeline]
    if panel.empty:
        ax.set_title(f"{display_name(pipeline)} -- no data")
        continue

    for model in MODEL_ORDER:
        col = MODEL_COLORS.get(model, "#666666")
        for P_val, marker in P_MARKERS.items():
            pts = panel[(panel["model"] == model) & (panel["P"] == P_val)]
            if pts.empty:
                continue
            lkey = (model, P_val)
            label = f"{model} -- P={P_val}" if lkey not in added_labels else ""
            ax.scatter(
                pts["mean_epoch"], pts["frac_subjects_correct"],
                color=col, marker=marker, s=55, alpha=0.82,
                edgecolors="black", linewidth=0.5, zorder=3,
                label=label,
            )
            if label:
                added_labels.add(lkey)

    # Compute axis limits first so diagonal spans the full viewport
    x_lo = min(panel["mean_epoch"].min() - 0.02, CHANCE_EPOCH - 0.03)
    x_hi = panel["mean_epoch"].max() + 0.02
    y_lo = min(panel["frac_subjects_correct"].min() - 0.02, CHANCE_SUBJ - 0.03)
    y_hi = max(panel["frac_subjects_correct"].max() + 0.02, CHANCE_SUBJ + 0.03)

    # Diagonal y=x — spans full axis range so it always crosses both axes
    diag_lo = min(x_lo, y_lo) - 0.01
    diag_hi = max(x_hi, y_hi) + 0.01
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", lw=1.0, alpha=0.40,
            label="y = x  (subject acc = epoch acc)" if ai == 0 else "")

    # Chance lines
    ax.axhline(CHANCE_SUBJ,  color="red", linestyle=":", linewidth=1.0, alpha=0.55,
               label="Chance" if ai == 0 else "")
    ax.axvline(CHANCE_EPOCH, color="red", linestyle=":", linewidth=1.0, alpha=0.55)

    ax.set_title(display_name(pipeline), fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch accuracy", fontsize=13)
    ax.set_ylabel("Subject accuracy", fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

# Shared legend — placed below both subplots
handles, labels = [], []
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l and l not in labels:
            handles.append(h)
            labels.append(l)
fig.tight_layout(pad=1.5)
fig.subplots_adjust(bottom=0.20)
fig.legend(
    handles, labels,
    loc="lower center", bbox_to_anchor=(0.5, 0.02),
    ncol=5, fontsize=11, framealpha=0.88, edgecolor="gray",
    handletextpad=0.3, columnspacing=1.0, labelspacing=0.3,
)
outfile = OUT_DIR / "ftd_trap3_epoch_vs_subject_P6_P2.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"\nSaved -> {outfile}")

# Print stats
print("\n=== FTD Trap 3 Stats ===")
for pipeline in ["PCA", "FTest"]:
    panel = agg_df[agg_df["pipeline"] == pipeline]
    for P_val in [6, 2]:
        print(f"\n{display_name(pipeline)} P={P_val}:")
        for model in MODEL_ORDER:
            pts = panel[(panel["model"] == model) & (panel["P"] == P_val)]
            if not pts.empty:
                print(f"  {model}: epoch={pts['mean_epoch'].values[0]:.4f}  subj={pts['frac_subjects_correct'].values[0]:.4f}")
