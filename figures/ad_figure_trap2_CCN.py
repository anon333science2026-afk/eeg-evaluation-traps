#!/usr/bin/env python3
"""
AD vs C — Trap 2: Variance vs Hold-Out Size (P=6 vs P=2).
2×2 grid: F-test top row, PCA bottom row; P=6 left, P=2 right.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
REPO = Path(__file__).resolve().parent.parent

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Helvetica"
matplotlib.use("Agg")
np.random.seed(42)

BASE       = Path(__file__).parent
DATA_CSV   = Path(f"{REPO}/data/results/ad/ad_all_experiments_combined.csv")
OUTDIR     = REPO / "figures/out"


MODEL_ORDER  = ["MLP", "XGBoost", "SVM", "KNN"]
MODEL_COLORS = {
    "MLP": "#1f77b4", "XGBoost": "#ff7f0e",
    "SVM": "#2ca02c", "KNN": "#d62728",
}

EXPERIMENT_MAP = {
    "ANOVA_P6": "ANOVA_L_6_Random",
    "ANOVA_P2": "ANOVA_L_2_Random",
    "PCA_P6":   "PCA_L_6_Random",
    "PCA_P2":   "PCA_L_2_Random",
}

CHANCE = 18607 / 34044   # epoch-level majority-class (control) ≈ 0.547


def load_best_hp(df_lpso, exp_name):
    sub    = df_lpso[df_lpso["experiment"] == exp_name]
    result = {}
    for model in MODEL_ORDER:
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        best_hp = m.groupby("hyperparams")["test_accuracy"].median().idxmax()
        vals    = m[m["hyperparams"] == best_hp]["test_accuracy"].tolist()
        result[model] = vals
        q1, q3 = np.percentile(vals, [25, 75])
        print(f"  {model}: n={len(vals)} median={np.median(vals)*100:.1f}% "
              f"IQR={((q3-q1)*100):.1f}pp range=[{min(vals)*100:.1f}%,{max(vals)*100:.1f}%]")
    return result


def single_panel(ax, data, title, ylim=(0.0, 1.0)):
    box_data, labels, positions = [], [], []
    for i, model in enumerate(MODEL_ORDER):
        if model in data:
            box_data.append(data[model])
            labels.append(model)
            positions.append(i)

    if box_data:
        bp = ax.boxplot(
            box_data, positions=positions, tick_labels=labels,
            patch_artist=True, showmeans=False, showfliers=False, widths=0.6,
        )
        for patch, model in zip(bp["boxes"], labels):
            patch.set_facecolor(MODEL_COLORS.get(model, "#666666"))
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.95)
        for el in ["whiskers", "medians", "caps"]:
            for item in bp[el]:
                item.set_color("black")
                item.set_linewidth(0.95)

        for i, (model, pos) in enumerate(zip(labels, positions)):
            vals      = np.array(box_data[i])
            mean_val  = np.mean(vals)
            distances = np.abs(vals - mean_val)
            max_d     = np.max(distances) if np.max(distances) > 0 else 1.0
            jit_sc    = 1.0 / (1.0 + 2.0 * (distances / max_d))
            jitter    = np.random.normal(0, 0.03, len(vals)) * jit_sc
            ax.scatter(pos + jitter, vals, alpha=0.5, s=15,
                       color=MODEL_COLORS.get(model, "#666666"),
                       edgecolors="black", linewidth=0.5, zorder=10)

    ax.axhline(y=CHANCE, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_ylabel("Epoch accuracy", fontsize=13)
    ax.set_ylim(ylim)
    ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=6)


def main():
    df   = pd.read_csv(DATA_CSV)
    lpso = df[df["experiment_type"] == "LPSO_Random_50"].copy()

    all_data = {}
    for exp_key, exp_name in EXPERIMENT_MAP.items():
        print(f"\n{exp_key}  ({exp_name}):")
        all_data[exp_key] = load_best_hp(lpso, exp_name)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.30})
    panels = {
        "ANOVA_P6": axes[0, 0], "ANOVA_P2": axes[0, 1],
        "PCA_P6":   axes[1, 0], "PCA_P2":   axes[1, 1],
    }
    titles = {
        "ANOVA_P6": "T-test — P=6",
        "ANOVA_P2": "T-test — P=2",
        "PCA_P6":   "PCA — P=6",
        "PCA_P2":   "PCA — P=2",
    }
    for key, ax in panels.items():
        single_panel(ax, all_data.get(key, {}), titles[key])

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    outfile = OUTDIR / "ad_trap2_variance_holdout_size_CCN.png"

    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
