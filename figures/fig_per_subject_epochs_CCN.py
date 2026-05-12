#!/usr/bin/env python3
"""
Per-subject epoch count and classification success rate — no suptitle.
Loads pre-computed per_subject_epochs_<disease>.csv files.
Adapted for NeurIPS figures output directory.
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]

CSV_DIR = Path(f"{REPO}/data/per_subject_summaries")
OUTDIR  = Path(f"{REPO}/figures/out")

CONFIGS = [
    {"name": "AD",     "csv": CSV_DIR / "per_subject_epochs_AD.csv",     "color_dis": "#d62728", "color_ctrl": "#1f77b4"},
    {"name": "FTD",    "csv": CSV_DIR / "per_subject_epochs_FTD.csv",    "color_dis": "#ff7f0e", "color_ctrl": "#1f77b4"},
    {"name": "MDD EC", "csv": CSV_DIR / "per_subject_epochs_MDD_EC.csv", "color_dis": "#2ca02c", "color_ctrl": "#1f77b4"},
    {"name": "MDD EO", "csv": CSV_DIR / "per_subject_epochs_MDD_EO.csv", "color_dis": "#9467bd", "color_ctrl": "#1f77b4"},
    {"name": "SCZ",    "csv": CSV_DIR / "per_subject_epochs_SCZ.csv",    "color_dis": "#8c564b", "color_ctrl": "#1f77b4"},
]

all_data = {}
for cfg in CONFIGS:
    if cfg["csv"].exists():
        all_data[cfg["name"]] = (pd.read_csv(cfg["csv"]), cfg)
    else:
        print(f"WARNING: {cfg['csv']} not found")

n_diseases = len(all_data)
fig, axes = plt.subplots(2, n_diseases, figsize=(6 * n_diseases, 10),
                          gridspec_kw={"hspace": 0.55, "wspace": 0.38,
                                       "height_ratios": [1.5, 1]})
if n_diseases == 1:
    axes = axes.reshape(2, 1)

for col_i, (name, (subj_df, config)) in enumerate(all_data.items()):
    ax_bar  = axes[0, col_i]
    ax_rate = axes[1, col_i]

    dis_df  = subj_df[subj_df["is_disease"]]
    ctrl_df = subj_df[~subj_df["is_disease"]]
    ordered = pd.concat([dis_df, ctrl_df], ignore_index=True)
    x = np.arange(len(ordered))
    colors = [config["color_dis"] if r else config["color_ctrl"]
              for r in ordered["is_disease"]]

    ax_bar.bar(x, ordered["n_epochs"], color=colors, alpha=0.75,
               edgecolor="none", width=0.85)
    n_dis = dis_df["SubjectID"].nunique()
    ax_bar.axvline(n_dis - 0.5, color="black", lw=1.2, ls="--", alpha=0.5)
    ax_bar.set_title(name, fontsize=14, fontweight="bold")
    ax_bar.set_ylabel("Test epochs (all folds)", fontsize=13)
    ax_bar.set_xlabel("Subject index", fontsize=13)
    ax_bar.tick_params(labelsize=13)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", alpha=0.2)

    correct   = ordered["success_rate"].values
    incorrect = 1.0 - correct
    bar_colors = [config["color_dis"] if r else config["color_ctrl"]
                  for r in ordered["is_disease"]]
    ax_rate.bar(x, correct,   color=bar_colors, alpha=0.75, width=0.85)
    ax_rate.bar(x, incorrect, bottom=correct, color="#dddddd", alpha=0.5, width=0.85)
    ax_rate.axhline(0.5, color="gray", lw=0.9, ls=":", alpha=0.7)
    ax_rate.axvline(n_dis - 0.5, color="black", lw=1.2, ls="--", alpha=0.5)
    ax_rate.set_ylim(0, 1.05)
    ax_rate.set_ylabel("Epoch success rate", fontsize=13)
    ax_rate.set_xlabel("Subject index", fontsize=13)
    ax_rate.tick_params(labelsize=13)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.grid(axis="y", alpha=0.2)

    if col_i == 0:
        dis_patch  = mpatches.Patch(color=config["color_dis"],  alpha=0.75, label="Disease")
        ctrl_patch = mpatches.Patch(color=config["color_ctrl"], alpha=0.75, label="Control")
        ax_bar.legend(handles=[dis_patch, ctrl_patch], fontsize=11, loc="upper right")

outfile = OUTDIR / "fig_per_subject_epochs.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {outfile}")
