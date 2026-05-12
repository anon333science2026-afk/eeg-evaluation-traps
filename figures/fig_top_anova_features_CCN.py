#!/usr/bin/env python3
"""
Top-10 EEG features by |Cohen's d| per disease — no suptitle.
Loads pre-computed CSVs from anova_feature_analysis/.
Adapted for NeurIPS figures output directory.
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]

CSV_DIR = Path(f"{REPO}/data/per_subject_summaries")
OUTDIR  = Path(f"{REPO}/figures/out")

DATASETS = [
    ("AD",     CSV_DIR / "top_anova_features_AD.csv",     "#1f77b4"),
    ("FTD",    CSV_DIR / "top_anova_features_FTD.csv",    "#ff7f0e"),
    ("MDD EC", CSV_DIR / "top_anova_features_MDD_EC.csv", "#2ca02c"),
    ("MDD EO", CSV_DIR / "top_anova_features_MDD_EO.csv", "#d62728"),
    ("SCZ",    CSV_DIR / "top_anova_features_SCZ.csv",    "#8c564b"),
]

N_TOP = 10
fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 5),
                          gridspec_kw={"wspace": 0.45})

for ax, (name, csv_path, color) in zip(axes, DATASETS):
    df = pd.read_csv(csv_path)
    top = df.head(N_TOP)
    ax.barh(range(N_TOP)[::-1], top["abs_d"].values,
            color=color, alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(N_TOP)[::-1])
    ax.set_yticklabels(top["label"].values, fontsize=8)
    ax.set_xlabel("|Cohen's d|", fontsize=9)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0.2, color="gray", lw=0.8, ls=":", alpha=0.6, label="small (0.2)")
    ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="medium (0.5)")
    ax.axvline(0.8, color="gray", lw=0.8, ls="-", alpha=0.4, label="large (0.8)")
    if ax == axes[0]:
        ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.2)

outfile = OUTDIR / "fig_top_anova_features.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {outfile}")
