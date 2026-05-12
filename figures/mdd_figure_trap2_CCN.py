#!/usr/bin/env python3
"""
MDD vs Control -- Trap 2: Lucky folds / IQR inflation.
Shows BOTH PCA and ANOVA at P=6 and P=2, for EC and EO.

Layout: 2 rows (EC top, EO bottom) x 4 columns (PCA P=6, PCA P=2, ANOVA P=6, ANOVA P=2)

Chance levels:
  EC epoch-level: 0.5148 (cntrl majority)
  EO epoch-level: 0.5061 (cntrl majority)

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

DATA_CSV  = Path(f"{REPO}/data/results/mdd/mdd_all_experiments_combined.csv")
OUTDIR    = Path(f"{REPO}/figures/out")

CHANCE_EC = 0.5033  # verified vs canonical parquet 2026-05-07
CHANCE_EO = 0.5199  # verified vs canonical parquet 2026-05-07

MODEL_ORDER  = ["MLP", "XGBoost", "SVM", "KNN"]
MODEL_COLORS = {
    "MLP":     "#1f77b4",
    "XGBoost": "#ff7f0e",
    "SVM":     "#2ca02c",
    "KNN":     "#d62728",
}

LAYOUT = [
    ("PCA_L_6_EC",   "PCA - P=6"),
    ("PCA_L_2_EC",   "PCA - P=2"),
    ("ANOVA_L_6_EC", "T-test - P=6"),
    ("ANOVA_L_2_EC", "T-test - P=2"),
]
LAYOUT_EO = [(e.replace("_EC", "_EO"), t) for e, t in LAYOUT]


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
                item.set_linewidth(0.95)
        for i, (model, pos) in enumerate(zip(labels, positions)):
            values = np.array(box_data[i])
            dist = np.abs(values - np.mean(values))
            max_d = np.max(dist) if len(dist) > 0 else 1.0
            scales = 1.0 / (1.0 + 2.0 * dist / max_d) if max_d > 0 else np.ones(len(values))
            jitter = np.random.normal(0, 0.025, len(values)) * scales
            ax.scatter(pos + jitter, values, alpha=0.45, s=10,
                       color=MODEL_COLORS[model], edgecolors="black",
                       linewidth=0.4, zorder=10)

    ax.axhline(chance, color="red", linestyle=":", alpha=0.6, linewidth=1.2,
               label=f"Chance ({chance:.3f})")
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

    fig, axes = plt.subplots(2, 4, figsize=(14, 9),
                             gridspec_kw={"hspace": 0.60, "wspace": 0.32})

    fs_labels  = ["PCA", "PCA", "T-test", "T-test"]
    p_labels   = ["P=6", "P=2", "P=6", "P=2"]

    for col_idx, ((exp_ec, _), (exp_eo, _)) in enumerate(zip(LAYOUT, LAYOUT_EO)):
        fs  = fs_labels[col_idx]
        plb = p_labels[col_idx]

        # EC row
        data_ec = load_best_hp(df, exp_ec)
        draw_panel(axes[0, col_idx], data_ec, CHANCE_EC)
        axes[0, col_idx].set_title(f"{fs} — {plb}\nEC (Eyes Closed)", fontsize=14, fontweight="bold", pad=6)
        if col_idx == 0:
            axes[0, col_idx].set_ylabel("Epoch accuracy", fontsize=13)

        # EO row
        data_eo = load_best_hp(df, exp_eo)
        draw_panel(axes[1, col_idx], data_eo, CHANCE_EO)
        axes[1, col_idx].set_title(f"{fs} — {plb}\nEO (Eyes Open)", fontsize=14, fontweight="bold", pad=6)
        if col_idx == 0:
            axes[1, col_idx].set_ylabel("Epoch accuracy", fontsize=13)

        # Print stats
        for cond, data, chance in [("EC", data_ec, CHANCE_EC), ("EO", data_eo, CHANCE_EO)]:
            print(f"\n{exp_ec if cond=='EC' else exp_eo}:")
            for model in MODEL_ORDER:
                if model not in data: continue
                vals = np.array(data[model])
                iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
                print(f"  {model}: median={np.median(vals):.3f}  IQR={iqr:.3f}  max={vals.max():.3f}  n={len(vals)}")

    from matplotlib.lines import Line2D
    ec_line = Line2D([0], [0], color="red", ls=":", lw=1.5, label=f"EC epoch chance ({CHANCE_EC:.3f})")
    eo_line = Line2D([0], [0], color="red", ls=":", lw=1.5, label=f"EO epoch chance ({CHANCE_EO:.3f})")
    fig.legend(handles=[ec_line, eo_line], loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=11, framealpha=0.9)

    outfile = OUTDIR / "mdd_trap2_CCN.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {outfile}")


if __name__ == "__main__":
    main()
