#!/usr/bin/env python3
"""
FTD vs C — Trap 2: Lucky folds / IQR inflation.
Shows BOTH PCA and ANOVA at P=6 and P=2.

Layout: 1 row x 4 columns (PCA P=6, PCA P=2, ANOVA P=6, ANOVA P=2)

Chance level:
  FTD epoch-level: 0.5422 (control majority)

Adapted for NeurIPS figures output directory.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]
np.random.seed(42)

DATA_CSV  = Path(f"{REPO}/processed_data/results/ftd/ftd_all_experiments_combined.csv")
OUTDIR    = Path(f"{REPO}/figures/out")

CHANCE_EPOCH = 0.5965  # epoch-level (control majority; verified vs canonical parquet 2026-05-07)

MODEL_ORDER  = ["MLP", "XGBoost", "SVM", "KNN"]
MODEL_COLORS = {
    "MLP":     "#1f77b4",
    "XGBoost": "#ff7f0e",
    "SVM":     "#2ca02c",
    "KNN":     "#d62728",
}

LAYOUT = [
    ("PCA_L_6_FTD",   "PCA - P=6"),
    ("PCA_L_2_FTD",   "PCA - P=2"),
    ("ANOVA_L_6_FTD", "T-test - P=6"),
    ("ANOVA_L_2_FTD", "T-test - P=2"),
]


def load_best_hp(df, exp_name):
    sub = df[df["experiment"] == exp_name]
    result = {}
    for model in MODEL_ORDER:
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        best_hp = m.groupby("hyperparams")["test_accuracy"].median().idxmax()
        vals = m[m["hyperparams"] == best_hp]["test_accuracy"].tolist()
        result[model] = vals
    return result


def draw_panel(ax, data, chance, ylim=(0.2, 1.05)):
    box_data, labels, positions = [], [], []
    for i, model in enumerate(MODEL_ORDER):
        if model in data:
            box_data.append(data[model])
            labels.append(model)
            positions.append(i)

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, tick_labels=labels,
                        patch_artist=True, showmeans=False, showfliers=False, widths=0.55)
        for patch, model in zip(bp["boxes"], labels):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.72)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.95)
        for element in ["whiskers", "medians", "caps"]:
            for item in bp[element]:
                item.set_color("black")
                item.set_linewidth(1.1)
        for i, (model, pos) in enumerate(zip(labels, positions)):
            values = np.array(box_data[i])
            dist = np.abs(values - np.mean(values))
            max_d = np.max(dist) if len(dist) > 0 else 1.0
            scales = 1.0 / (1.0 + 2.0 * dist / max_d) if max_d > 0 else np.ones(len(values))
            jitter = np.random.normal(0, 0.025, len(values)) * scales
            ax.scatter(pos + jitter, values, alpha=0.45, s=10,
                       color=MODEL_COLORS[model], edgecolors="black",
                       linewidth=0.4, zorder=10)

    ax.axhline(chance, color="red", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df):,} rows")

    fig, axes = plt.subplots(1, 4, figsize=(14, 5),
                             gridspec_kw={"wspace": 0.32})

    col_titles = ["PCA - P=6", "PCA - P=2", "T-test - P=6", "T-test - P=2"]

    for col_idx, (exp_name, _) in enumerate(LAYOUT):
        data = load_best_hp(df, exp_name)
        draw_panel(axes[col_idx], data, CHANCE_EPOCH)
        axes[col_idx].set_title(col_titles[col_idx], fontsize=14, fontweight="bold", pad=6)
        if col_idx == 0:
            axes[col_idx].set_ylabel("Epoch accuracy", fontsize=13)

        # Print stats
        print(f"\n{exp_name}:")
        for model in MODEL_ORDER:
            if model not in data:
                continue
            vals = np.array(data[model])
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            print(f"  {model}: median={np.median(vals):.3f}  IQR={iqr*100:.1f}pp  max={vals.max():.3f}  n={len(vals)}")

    # IQR ratios
    print("\n=== IQR ratios ANOVA P=2 / P=6 ===")
    p6_iqrs = {}
    p2_iqrs = {}
    for model in MODEL_ORDER:
        sub6 = df[(df["experiment"] == "ANOVA_L_6_FTD") & (df["model"] == model)]
        sub2 = df[(df["experiment"] == "ANOVA_L_2_FTD") & (df["model"] == model)]
        if sub6.empty or sub2.empty:
            continue
        best6 = sub6.groupby("hyperparams")["test_accuracy"].median().idxmax()
        best2 = sub2.groupby("hyperparams")["test_accuracy"].median().idxmax()
        v6 = sub6[sub6["hyperparams"] == best6]["test_accuracy"].values
        v2 = sub2[sub2["hyperparams"] == best2]["test_accuracy"].values
        iqr6 = np.percentile(v6, 75) - np.percentile(v6, 25)
        iqr2 = np.percentile(v2, 75) - np.percentile(v2, 25)
        p6_iqrs[model] = iqr6
        p2_iqrs[model] = iqr2
        ratio = iqr2 / iqr6 if iqr6 > 0 else float("nan")
        print(f"  {model}: P6={iqr6*100:.1f}pp  P2={iqr2*100:.1f}pp  ratio={ratio:.1f}x")

    from matplotlib.lines import Line2D
    chance_line = Line2D([0], [0], color="red", ls=":", lw=1.5,
                         label=f"Epoch chance ({CHANCE_EPOCH:.3f})")
    fig.legend(handles=[chance_line], loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=1, fontsize=11, framealpha=0.9)

    outfile = OUTDIR / "ftd_trap2_CCN.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {outfile}")


if __name__ == "__main__":
    main()
