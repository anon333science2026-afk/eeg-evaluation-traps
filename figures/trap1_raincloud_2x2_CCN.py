#!/usr/bin/env python3
"""
Trap 1 raincloud 2x2 — KNN locked to k=7.
No suptitle. Updated color scheme: overlap=deep purple, disjoint=teal.
Adapted for NeurIPS figures output directory.
"""

import json
import re
import yaml
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

BASE      = Path(f"{REPO}/processed_data/ad_extras")
INTRA_DIR = BASE / "intra-subject"
DATA_CSV  = BASE / "all_experiments_combined.csv"
OUTDIR    = Path(f"{REPO}/figures/out")

COL = {
    "fingerprint": "#e07b39",   # orange — fingerprinting (subject-ID)
    "overlap":     "#7b2d8b",   # deep purple — subject-overlap
    "disjoint":    "#2a9d8f",   # teal — subject-disjoint (LPSO)
}
MODEL_ORDER = ["MLP", "XGBoost", "SVM", "KNN"]

MODEL_NAME_MAP = {
    "MLP (Neural Network)": "MLP",
    "MLP_(Neural_Network)": "MLP",
    "MLP": "MLP",
    "XGBoost": "XGBoost",
    "SVM": "SVM",
    "KNN": "KNN",
}

LOCKED_KNN_K = 7
N_SUBJECTS     = 65
N_AD           = 36
CHANCE_FP      = 1.0 / N_SUBJECTS
CHANCE_DISEASE = N_AD / N_SUBJECTS


def _norm(name: str) -> str:
    return MODEL_NAME_MAP.get(name.replace("_", " ").strip(), name)


def extract_per_version(exp_name: str) -> dict:
    exp_path = INTRA_DIR / exp_name
    results  = defaultdict(list)
    if not exp_path.exists():
        print(f"  WARNING: {exp_path} not found")
        return results

    skip_versions = {"ANOVA_W_F_v0"}
    version_dirs = sorted(
        [d for d in exp_path.iterdir()
         if d.is_dir() and re.match(rf"{re.escape(exp_name)}_v\d+$", d.name)
         and d.name not in skip_versions],
        key=lambda d: int(d.name.split("_v")[-1]),
    )
    print(f"  {exp_name}: {len(version_dirs)} versions found")

    for vdir in version_dirs:
        ml_path = vdir / "ml_results_grid_search"
        if not ml_path.exists():
            continue
        version_best: dict = {}
        for model_dir in ml_path.iterdir():
            if not model_dir.is_dir() or model_dir.name in {"graphs", "debug"}:
                continue
            model = _norm(model_dir.name)
            if model not in MODEL_ORDER:
                continue
            if model == "KNN":
                knn_acc = _get_knn_k7_accuracy(model_dir, vdir)
                if knn_acc is not None:
                    version_best[model] = knn_acc
            else:
                best_acc = None
                for rf in model_dir.rglob("results.json"):
                    try:
                        data = json.loads(rf.read_text())
                        acc  = (data.get("test_results", {}).get("accuracy")
                                or data.get("test_accuracy")
                                or data.get("detailed_results", {}).get("test_accuracy"))
                        if acc is not None:
                            acc = float(acc)
                            if best_acc is None or acc > best_acc:
                                best_acc = acc
                    except Exception:
                        pass
                if best_acc is not None:
                    version_best[model] = best_acc
        for model, acc in version_best.items():
            results[model].append(acc)

    for model in MODEL_ORDER:
        vals = results.get(model, [])
        print(f"    {model}: n={len(vals)}")
    return results


def _get_knn_k7_accuracy(knn_dir: Path, vdir: Path):
    hp_csv = knn_dir / "hyperparameter_comparison.csv"
    if hp_csv.exists():
        try:
            df = pd.read_csv(hp_csv)
            for col in df.columns:
                if "n_neighbor" in col.lower():
                    row = df[df[col] == LOCKED_KNN_K]
                    if not row.empty:
                        for acc_col in ["test_accuracy", "accuracy", "mean_test_accuracy"]:
                            if acc_col in row.columns:
                                return float(row.iloc[0][acc_col])
        except Exception:
            pass
    for rf in knn_dir.rglob("results.json"):
        try:
            data = json.loads(rf.read_text())
            hp = data.get("hyperparameters", data.get("best_params", {}))
            if hp.get("n_neighbors") == LOCKED_KNN_K:
                acc = (data.get("test_results", {}).get("accuracy")
                       or data.get("test_accuracy")
                       or data.get("detailed_results", {}).get("test_accuracy"))
                if acc is not None:
                    return float(acc)
        except Exception:
            pass
    best_yaml = knn_dir / "best_results.yaml"
    if best_yaml.exists():
        try:
            d = yaml.safe_load(best_yaml.read_text())
            if d.get("best_hyperparams", {}).get("n_neighbors") == LOCKED_KNN_K:
                return float(d["best_metrics"]["test_accuracy"])
        except Exception:
            pass
    return None


print("Loading within-subject data (KNN locked to k=7) …")
data_ws = {}
for exp in ["PCA_W_F", "PCA_W_C", "ANOVA_W_F", "ANOVA_W_C"]:
    data_ws[exp] = extract_per_version(exp)

raw = pd.read_csv(DATA_CSV)
_dj = raw[(raw["experiment_type"] == "LPSO_Random_50") & (raw["holdout_size_P"] == 6)]
KNN_K7_HP = '{"metric": "euclidean", "n_neighbors": 7, "weights": "uniform"}'


def best_hp_disjoint_knn7(fs: str) -> dict:
    sub = _dj[_dj["feature_set"] == fs]
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


dj_pca   = best_hp_disjoint_knn7("PCA")
dj_anova = best_hp_disjoint_knn7("ANOVA")


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


def draw_panel(ax, left_vals, right_vals, col_L, col_R,
               bw_L=0.45, bw_R=0.3, chance=None):
    for mi, model in enumerate(MODEL_ORDER):
        xi = mi
        L  = np.asarray(left_vals.get(model, []))
        R  = np.asarray(right_vals.get(model, []))

        half_violin(ax, L, xi - 0.02, color=col_L, side="left",  bw=bw_L)
        if len(L) > 0:
            med = np.median(L)
            dist = np.abs(L - med)
            max_d = dist.max() if dist.max() > 0 else 1.0
            width = 0.25 * (1.0 - dist / max_d) ** 1.5
            jit = xi - 0.05 - np.random.uniform(0, 1, len(L)) * width
            ax.scatter(jit, L, color=col_L, s=18, alpha=0.55, linewidths=0)
            ax.plot([xi - 0.34, xi - 0.03], [med] * 2, color=col_L, linewidth=2.5)

        half_violin(ax, R, xi + 0.02, color=col_R, side="right", bw=bw_R)
        if len(R) > 0:
            med = np.median(R)
            dist = np.abs(R - med)
            max_d = dist.max() if dist.max() > 0 else 1.0
            width = 0.25 * (1.0 - dist / max_d) ** 1.5
            jit = xi + 0.05 + np.random.uniform(0, 1, len(R)) * width
            ax.scatter(jit, R, color=col_R, s=18, alpha=0.55, linewidths=0)
            ax.plot([xi + 0.03, xi + 0.34], [med] * 2, color=col_R, linewidth=2.5)

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xlim(-0.65, len(MODEL_ORDER) - 0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if chance is not None:
        ax.axhline(chance, color="red", linewidth=1.2, linestyle=":", alpha=0.7)


LEGEND_FP = mpatches.Patch(facecolor=COL["fingerprint"], label="Fingerprinting (subject-ID)")
LEGEND_OV = mpatches.Patch(facecolor=COL["overlap"],     label="Subject-overlap")
LEGEND_DJ = mpatches.Patch(facecolor=COL["disjoint"],    label="Subject-disjoint (LPSO P=6)")

fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharey=True, sharex="col",
                         gridspec_kw={"hspace": 0.38, "wspace": 0.06})

for row_i, (fs_label, fs_key, dj) in enumerate([
    ("PCA",    "PCA",   dj_pca),
    ("T-test", "ANOVA", dj_anova),
]):
    fp = {m: np.asarray(data_ws[f"{fs_key}_W_F"][m]) for m in MODEL_ORDER}
    ov = {m: np.asarray(data_ws[f"{fs_key}_W_C"][m]) for m in MODEL_ORDER}

    ax = axes[row_i][0]
    draw_panel(ax, fp, ov, COL["fingerprint"], COL["overlap"],
               bw_L=0.45, bw_R=0.45, chance=CHANCE_FP)
    ax.set_title(f"{fs_label} — Within-subject splits\nFingerprinting  vs  Subject-overlap",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Epoch test accuracy", fontsize=13)
    ax.legend(handles=[LEGEND_FP, LEGEND_OV], fontsize=11, loc="lower right")

    ax = axes[row_i][1]
    draw_panel(ax, ov, dj, COL["overlap"], COL["disjoint"],
               bw_L=0.45, bw_R=0.3, chance=CHANCE_DISEASE)
    ax.set_title(f"{fs_label} — Disease classification\nSubject-overlap  vs  Subject-disjoint (LPSO P=6)",
                 fontsize=14, fontweight="bold")
    ax.legend(handles=[LEGEND_OV, LEGEND_DJ], fontsize=11, loc="lower right")

plt.tight_layout()
outfile = OUTDIR / "trap1_knn=7_raincloud_v11_2x2.png"
fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved -> {outfile}")
