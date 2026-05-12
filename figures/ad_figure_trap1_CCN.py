#!/usr/bin/env python3
"""
AD vs C — Trap 1: subject-disjoint (LPSO P=6) vs subject-overlap (W_C).
Vertical 2×1: F-test (top), PCA (bottom). KNN locked to k=7.

Dataset: AD vs Control (reject=800)
  AD:  36 subjects (sub-001 to sub-036)
  Control: 29 subjects (sub-037 to sub-065)
  Total: 65 subjects
  Chance disease (majority=AD): 36/65 ≈ 0.554
"""

import json
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
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Helvetica"
np.random.seed(42)

BASE     = Path(__file__).parent
DATA_DIR = Path(f"{REPO}/processed_data/results/ad")
ROOT     = REPO
DATA_CSV = DATA_DIR / "ad_all_experiments_combined.csv"
WC_DIR   = ROOT   # W_C result dirs live at the root
OUTDIR   = REPO / "figures/out"


MODEL_ORDER  = ["MLP", "XGBoost", "SVM", "KNN"]
LOCKED_KNN_K = 7
N_SUBJECTS   = 65
N_AD         = 36
CHANCE       = 18607 / 34044   # epoch-level majority-class (control) ≈ 0.547

COL = {"overlap": "#7b2d8b", "disjoint": "#2a9d8f"}

MODEL_NORMALIZE = {
    "KNN": "KNN", "SVM": "SVM", "XGBoost": "XGBoost",
    "MLP_(Neural_Network)": "MLP", "MLP (Neural Network)": "MLP",
}


def _norm(name):
    return MODEL_NORMALIZE.get(name.strip(), name)


# ── Load W_C within-subject overlap data (KNN locked to k=7) ─────────────────
def load_wc(prefix):
    results = {m: [] for m in MODEL_ORDER}
    for seed in range(42, 52):
        exp_dir = WC_DIR / f"{prefix}_ad_cntrl_seed{seed}" / "ml_results_grid_search"
        if not exp_dir.exists():
            continue
        for model_dir in exp_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "graphs":
                continue
            model = _norm(model_dir.name)
            if model not in MODEL_ORDER:
                continue
            if model == "KNN":
                hp_csv = model_dir / "hyperparameter_comparison.csv"
                if hp_csv.exists():
                    df = pd.read_csv(hp_csv)
                    row = df[df["hyperparam_n_neighbors"] == LOCKED_KNN_K]
                    if not row.empty:
                        results["KNN"].append(float(row.iloc[0]["mean_test_accuracy"]))
                        continue
            # Other models: pick best accuracy
            best = None
            for rj_path in model_dir.rglob("results.json"):
                try:
                    rj  = json.loads(rj_path.read_text())
                    acc = (rj.get("test_results", {}).get("accuracy")
                           or rj.get("test_accuracy"))
                    if acc is not None and (best is None or float(acc) > best):
                        best = float(acc)
                except Exception:
                    pass
            if best is not None:
                results[model].append(best)
    for m, v in results.items():
        print(f"  {prefix} {m}: n={len(v)}"
              + (f"  median={np.median(v)*100:.2f}%" if v else ""))
    return results


# ── Load LPSO disjoint data ───────────────────────────────────────────────────
KNN_K7_HP = '{"metric": "euclidean", "n_neighbors": 7, "weights": "uniform"}'

def load_disjoint(feature_set):
    raw = pd.read_csv(DATA_CSV)
    sub = raw[(raw["experiment_type"] == "LPSO_Random_50")
              & (raw["holdout_size_P"] == 6)
              & (raw["feature_set"] == feature_set)]
    out = {}
    for model in MODEL_ORDER:
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        if model == "KNN":
            k7 = m[m["hyperparams"] == KNN_K7_HP]
            if not k7.empty:
                out[model] = k7["test_accuracy"].values
        else:
            top = m.groupby("hyperparams")["test_accuracy"].median().idxmax()
            out[model] = m[m["hyperparams"] == top]["test_accuracy"].values
    return out


# ── Drawing helpers ───────────────────────────────────────────────────────────
def half_violin(ax, vals, x, width=0.32, color="steelblue", alpha=0.65,
                side="right", bw=0.35):
    if len(vals) < 4:
        return
    kde     = gaussian_kde(vals, bw_method=bw)
    y_range = np.linspace(max(0.0, vals.min() - 0.01),
                          min(1.0, vals.max() + 0.01), 200)
    density = kde(y_range)
    density[density < 0.03 * density.max()] = 0.0  # clip hairline tails
    density = density / density.max() * width
    if side == "right":
        ax.fill_betweenx(y_range, x, x + density, color=color, alpha=alpha)
    else:
        ax.fill_betweenx(y_range, x - density, x, color=color, alpha=alpha)


def draw_panel(ax, disjoint_vals, overlap_vals, col_dj, col_ov,
               bw_dj=0.3, bw_ov=0.45, chance=None):
    for mi, model in enumerate(MODEL_ORDER):
        xi = mi
        # overlap on the LEFT, disjoint on the RIGHT
        OV = np.asarray(overlap_vals.get(model, []))
        DJ = np.asarray(disjoint_vals.get(model, []))

        if len(OV) > 0:
            half_violin(ax, OV, xi - 0.02, color=col_ov, side="left", bw=bw_ov)
            med = np.median(OV)
            dist = np.abs(OV - med)
            max_d = dist.max() if dist.max() > 0 else 1.0
            width = 0.25 * (1.0 - dist / max_d) ** 1.5  # narrow toward extremes
            jit = xi - 0.05 - np.random.uniform(0, 1, len(OV)) * width
            ax.scatter(jit, OV, color=col_ov, s=18, alpha=0.55, linewidths=0)
            ax.plot([xi - 0.34, xi - 0.03], [med] * 2,
                    color=col_ov, linewidth=2.5)

        if len(DJ) > 0:
            half_violin(ax, DJ, xi + 0.02, color=col_dj, side="right", bw=bw_dj)
            med = np.median(DJ)
            dist = np.abs(DJ - med)
            max_d = dist.max() if dist.max() > 0 else 1.0
            width = 0.25 * (1.0 - dist / max_d) ** 1.5
            jit = xi + 0.05 + np.random.uniform(0, 1, len(DJ)) * width
            ax.scatter(jit, DJ, color=col_dj, s=18, alpha=0.55, linewidths=0)
            ax.plot([xi + 0.03, xi + 0.34], [med] * 2,
                    color=col_dj, linewidth=2.5)

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=13)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-0.55, len(MODEL_ORDER) - 0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if chance is not None:
        ax.axhline(chance, color="red", linewidth=1.2, linestyle=":", alpha=0.7)


LEGEND_DJ = mpatches.Patch(facecolor=COL["disjoint"], label="Subject-disjoint (LPSO P=6)")
LEGEND_OV = mpatches.Patch(facecolor=COL["overlap"],  label="Subject-overlap")


# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading W_C overlap data …")
wc_anova = load_wc("ANOVA_W_C")
wc_pca   = load_wc("PCA_W_C")

print("\nLoading LPSO disjoint data …")
dj_anova = load_disjoint("ANOVA")
dj_pca   = load_disjoint("PCA")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_L, ax_R) = plt.subplots(
    1, 2, figsize=(12, 5.5),
    gridspec_kw={"wspace": 0.32},
)

ov_pca = {m: np.asarray(wc_pca[m]) for m in MODEL_ORDER}
draw_panel(ax_L, dj_pca, ov_pca, COL["disjoint"], COL["overlap"],
           bw_dj=0.3, bw_ov=0.45, chance=CHANCE)
ax_L.set_title("PCA", fontsize=14, fontweight="bold", pad=8)
ax_L.set_ylabel("Epoch accuracy", fontsize=13)
ax_L.legend(handles=[LEGEND_OV, LEGEND_DJ], fontsize=11, loc="lower right",
            framealpha=0.9)

ov_anova = {m: np.asarray(wc_anova[m]) for m in MODEL_ORDER}
draw_panel(ax_R, dj_anova, ov_anova, COL["disjoint"], COL["overlap"],
           bw_dj=0.3, bw_ov=0.45, chance=CHANCE)
ax_R.set_title("T-test", fontsize=14, fontweight="bold", pad=8)
ax_R.set_ylabel("Epoch accuracy", fontsize=13)
ax_R.legend(handles=[LEGEND_OV, LEGEND_DJ], fontsize=11, loc="lower right",
            framealpha=0.9)

fig.tight_layout(pad=1.5)
outfile = OUTDIR / "ad_trap1_disjoint_vs_overlap_CCN.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {outfile}")
