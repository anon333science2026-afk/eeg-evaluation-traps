#!/usr/bin/env python3
"""
SCZ vs C -- Trap 1 figure (both PCA and ANOVA, KNN=7).
1x2 layout: Left = PCA, Right = ANOVA
Each panel: LPSO disjoint (blue) vs subject-overlap W_C (red).

Dataset: SCZ vs Control
  SCZ: 14 subjects | Control: 14 subjects | Total: 28
  Chance (balanced): 0.500

Adapted for NeurIPS figures output directory.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
REPO = Path(__file__).resolve().parent.parent

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]
np.random.seed(42)

BASE_SCZ  = Path(f"{REPO}/processed_data/results/scz")
DATA_CSV  = BASE_SCZ / "scz_all_experiments_combined.csv"
OUTDIR    = Path(f"{REPO}/figures/out")

MODEL_ORDER  = ["MLP", "XGBoost", "SVM", "KNN"]
LOCKED_KNN_K = 7

CHANCE_EPOCH = 0.5353  # epoch-level (SCZ majority — recordings longer than control); verified vs canonical parquet 2026-05-07
SEEDS = list(range(42, 52))

COL_DIS = "#7b2d8b"   # deep purple — subject-overlap
COL_DJ  = "#2a9d8f"   # teal        — subject-disjoint (LPSO)


def load_wc_direct(feature: str) -> dict:
    """Load W_C subject-overlap accuracy from hyperparameter_comparison.csv."""
    results = defaultdict(list)
    for seed in SEEDS:
        dname = f"{feature}_W_C_scz_cntrl_seed{seed}"
        ml_dir = BASE_SCZ / dname / "ml_results_grid_search"
        if not ml_dir.exists():
            continue
        for model in MODEL_ORDER:
            mname = "MLP_(Neural_Network)" if model == "MLP" else model
            mdir  = ml_dir / mname
            if not mdir.is_dir():
                continue
            hp_csv = mdir / "hyperparameter_comparison.csv"
            if not hp_csv.exists():
                continue
            df = pd.read_csv(hp_csv)
            if model == "KNN":
                k_col = next((c for c in df.columns if "neighbor" in c.lower()), None)
                if k_col:
                    row = df[df[k_col].astype(str) == str(LOCKED_KNN_K)]
                    if not row.empty:
                        acc_col = next((c for c in df.columns if "accuracy" in c.lower()), None)
                        if acc_col:
                            results[model].append(float(row[acc_col].iloc[0]))
                continue
            acc_col = next((c for c in df.columns if "accuracy" in c.lower()), None)
            if acc_col:
                results[model].append(float(df[acc_col].max()))
    return dict(results)


def load_disjoint(feature: str) -> dict:
    """Load LPSO P=6 best-HP per model from compiled CSV."""
    df = pd.read_csv(DATA_CSV)
    exp_name = f"{feature}_L_6_SCZ"
    sub = df[df["experiment"] == exp_name]
    out = {}
    KNN_K7_HP = '{"metric": "euclidean", "n_neighbors": 7, "weights": "uniform"}'
    for model in MODEL_ORDER:
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        if model == "KNN":
            k7 = m[m["hyperparams"] == KNN_K7_HP]
            if not k7.empty:
                out[model] = k7["test_accuracy"].values
            continue
        best_hp = m.groupby("hyperparams")["test_accuracy"].median().idxmax()
        out[model] = m[m["hyperparams"] == best_hp]["test_accuracy"].values
    return out


def half_violin(ax, vals, x, width=0.30, color="steelblue", alpha=0.65,
                side="right", bw=0.35):
    vals = np.asarray(vals)
    if len(vals) < 3:
        return
    kde = gaussian_kde(vals, bw_method=bw)
    y_range = np.linspace(max(0.0, vals.min() - 0.01),
                          min(1.0, vals.max() + 0.01), 200)
    density = kde(y_range)
    density[density < 0.03 * density.max()] = 0.0  # clip hairline tails
    density = density / density.max() * width
    if side == "right":
        ax.fill_betweenx(y_range, x, x + density, color=color, alpha=alpha)
    else:
        ax.fill_betweenx(y_range, x - density, x, color=color, alpha=alpha)


def draw_panel(ax, left_vals, right_vals, col_L, col_R,
               chance_line, ylim=(0.0, 1.0), bw_L=0.3, bw_R=0.4):
    for mi, model in enumerate(MODEL_ORDER):
        L = np.asarray(left_vals.get(model, []))
        R = np.asarray(right_vals.get(model, []))
        xi = mi

        # overlap on LEFT, disjoint on RIGHT — funnel jitter
        half_violin(ax, L, xi - 0.02, color=col_L, side="left", bw=bw_L)
        if len(L) > 0:
            med = np.median(L); dist = np.abs(L - med)
            max_d = dist.max() if dist.max() > 0 else 1.0
            width = 0.25 * (1.0 - dist / max_d) ** 1.5
            jit = xi - 0.05 - np.random.uniform(0, 1, len(L)) * width
            ax.scatter(jit, L, color=col_L, s=18, alpha=0.55, linewidths=0)
            ax.plot([xi - 0.34, xi - 0.03], [med] * 2, color=col_L, linewidth=2.5)

        half_violin(ax, R, xi + 0.02, color=col_R, side="right", bw=bw_R)
        if len(R) > 0:
            med = np.median(R); dist = np.abs(R - med)
            max_d = dist.max() if dist.max() > 0 else 1.0
            width = 0.25 * (1.0 - dist / max_d) ** 1.5
            jit = xi + 0.05 + np.random.uniform(0, 1, len(R)) * width
            ax.scatter(jit, R, color=col_R, s=18, alpha=0.55, linewidths=0)
            ax.plot([xi + 0.03, xi + 0.34], [med] * 2, color=col_R, linewidth=2.5)

    ax.axhline(chance_line, color="red", lw=1.2, ls=":", alpha=0.7)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.55, len(MODEL_ORDER) - 0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")


# Load data
print("Loading W_C overlap data ...")
pca_wc   = load_wc_direct("PCA")
anova_wc = load_wc_direct("ANOVA")

print("Loading LPSO P=6 disjoint data ...")
dj_pca   = load_disjoint("PCA")
dj_anova = load_disjoint("ANOVA")

for m in MODEL_ORDER:
    print(f"  {m}: WC_PCA={len(pca_wc.get(m,[]))}  WC_ANOVA={len(anova_wc.get(m,[]))}  "
          f"DJ_PCA={len(dj_pca.get(m,[]))}  DJ_ANOVA={len(dj_anova.get(m,[]))}")


# Figure: 1x2 (cols=PCA/ANOVA)
LEG_DIS = mpatches.Patch(facecolor=COL_DIS, label="Subject-overlap")
LEG_DJ  = mpatches.Patch(facecolor=COL_DJ,  label="Subject-disjoint (LPSO P=6)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                         gridspec_kw={"wspace": 0.26})

panels = [
    (axes[0], dj_pca,   pca_wc,   CHANCE_EPOCH, (0.3, 1.0), "PCA"),
    (axes[1], dj_anova, anova_wc, CHANCE_EPOCH, (0.3, 1.0), "T-test"),
]

for ax, dj, wc, chance, ylim, title in panels:
    draw_panel(ax, wc, dj, COL_DIS, COL_DJ, chance_line=chance, ylim=ylim)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.legend(handles=[LEG_DIS, LEG_DJ], fontsize=11, loc="lower right")

axes[0].set_ylabel("Epoch accuracy", fontsize=13)

outfile = OUTDIR / "scz_trap1_CCN.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved -> {outfile}")

# Print stats
print("\n=== SCZ Trap 1 Stats ===")
for label, data in [("LPSO PCA", dj_pca), ("LPSO ANOVA", dj_anova),
                    ("WC PCA", pca_wc), ("WC ANOVA", anova_wc)]:
    print(f"\n{label}:")
    for m in MODEL_ORDER:
        v = np.array(data.get(m, []))
        if len(v) > 0:
            print(f"  {m}: median={np.median(v):.3f}  IQR={np.percentile(v,75)-np.percentile(v,25):.3f}  n={len(v)}")
