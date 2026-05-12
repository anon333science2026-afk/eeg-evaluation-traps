#!/usr/bin/env python3
"""
Subject-level accuracy small multiples P=2 — T-test label, no suptitle, 0.95 linewidth.
Loads fold_epoch_vs_subject_merged.csv from paper_subject_eval_outputs.
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

DATA_CSV = Path(f"{REPO}/processed_data/ad_extras/fold_epoch_vs_subject_merged.csv")
OUTDIR   = Path(f"{REPO}/figures/out")

MODEL_ORDER = ["MLP", "KNN", "SVM", "XGBoost"]
EXP_MODEL_KEYS = ["experiment", "pipeline", "strategy", "P", "model"]
MODEL_COLORS = {
    "MLP":     "#1f77b4",
    "XGBoost": "#ff7f0e",
    "SVM":     "#2ca02c",
    "KNN":     "#d62728",
}


def display_pipeline_name(pipeline: str) -> str:
    return "T-test" if pipeline == "FTest" else pipeline


def select_best_hp(df, criterion="epoch"):
    acc_col = "epoch_acc" if criterion == "epoch" else "subject_acc"
    rows = []
    for _k, grp in df.groupby(EXP_MODEL_KEYS):
        best_hp = grp.groupby("hyperparams")[acc_col].median().idxmax()
        best_rows = grp[grp["hyperparams"] == best_hp].copy()
        best_rows["best_hp"] = best_hp
        rows.append(best_rows)
    return pd.concat(rows, ignore_index=True)


def _boxplot_panel(ax, data_by_model, model_order, title, ylabel, P=6):
    for i, model in enumerate(model_order):
        vals_frac = np.array(data_by_model.get(model, []))
        if len(vals_frac) == 0:
            continue
        vals = vals_frac * P
        col  = MODEL_COLORS.get(model, "#666666")
        bp = ax.boxplot(
            vals, positions=[i], widths=0.6,
            patch_artist=True, showmeans=False, showfliers=False,
            medianprops=dict(color="black", linewidth=0.95),
            whiskerprops=dict(color="black", linewidth=0.95),
            capprops=dict(color="black", linewidth=0.95),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.95)

        mean_val  = np.mean(vals)
        distances = np.abs(vals - mean_val)
        max_d = np.max(distances) if np.max(distances) > 0 else 1.0
        jit_scales = 1.0 / (1.0 + 2.0 * (distances / max_d))
        jitter = np.random.normal(0, 0.03, len(vals)) * jit_scales
        ax.scatter(i + jitter, vals, alpha=0.5, s=20, color=col,
                   edgecolors="black", linewidth=0.5, zorder=10)

    ax.axhline(P / 2, color="gray", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, len(model_order) - 0.5)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(-0.4, P + 0.4)
    ax.set_yticks(range(P + 1))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_small_multiples(best_df, P_filter, out_path):
    sub = best_df[best_df["P"] == P_filter]
    strategies = [s for s in ["Uniform-12", "Random-50"] if s in sub["strategy"].unique()]
    pipelines  = [p for p in ["FTest", "PCA"] if p in sub["pipeline"].unique()]

    side_by_side = (len(strategies) == 1 and len(pipelines) == 2)

    if side_by_side:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = len(pipelines), len(strategies)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)

    if side_by_side:
        strategy = strategies[0]
        for ci, pipeline in enumerate(pipelines):
            panel = sub[(sub["pipeline"] == pipeline) & (sub["strategy"] == strategy)]
            data_by_model = {m: panel.loc[panel["model"] == m, "subject_acc"].values
                             for m in MODEL_ORDER}
            _boxplot_panel(
                axes[0][ci], data_by_model, MODEL_ORDER,
                title=f"{display_pipeline_name(pipeline)}  —  {strategy}",
                ylabel=(f"Subjects correctly classified (of {P_filter})" if ci == 0 else ""),
                P=P_filter,
            )
    else:
        for ri, pipeline in enumerate(pipelines):
            for ci, strategy in enumerate(strategies):
                panel = sub[(sub["pipeline"] == pipeline) & (sub["strategy"] == strategy)]
                data_by_model = {m: panel.loc[panel["model"] == m, "subject_acc"].values
                                 for m in MODEL_ORDER}
                _boxplot_panel(
                    axes[ri][ci], data_by_model, MODEL_ORDER,
                    title=f"{display_pipeline_name(pipeline)}  —  {strategy}",
                    ylabel=(f"Subjects correctly classified (of {P_filter})" if ci == 0 else ""),
                    P=P_filter,
                )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    np.random.seed(42)
    merged = pd.read_csv(DATA_CSV)
    print(f"Loaded: {len(merged):,} rows")
    best = select_best_hp(merged, "epoch")

    make_small_multiples(best, 2, OUTDIR / "fig_subject_accuracy_small_multiples_P2.png")
