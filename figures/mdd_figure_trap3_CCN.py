#!/usr/bin/env python3
"""
MDD vs Control -- Trap 3: Epoch vs Subject accuracy scatter.
2x2 layout: rows = EC/EO, columns = PCA/ANOVA.
P=6 = circles, P=2 = triangles.

Dataset: MDD vs Control
  MDD: 29 subjects / Control: 27 subjects / Total: 56
  Epoch-level chance: EC=0.5148, EO=0.5061 (cntrl epoch-majority)
  Subject-level chance: 29/56 = 0.5179 (MDD subject-majority)

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

IN_CSV  = Path(f"{REPO}/processed_data/results/mdd/mdd_fold_epoch_vs_subject.csv")
OUTDIR  = Path(f"{REPO}/figures/out")

MODEL_ORDER  = ["MLP", "XGBoost", "SVM", "KNN"]
MODEL_COLORS = {
    "MLP":     "#1f77b4",
    "XGBoost": "#ff7f0e",
    "SVM":     "#2ca02c",
    "KNN":     "#d62728",
}
P_MARKERS = {6: "o", 2: "^"}

CHANCE_EPOCH = {"EC": 0.5033, "EO": 0.5199}  # verified vs canonical parquet 2026-05-07
CHANCE_SUBJ  = 29 / 56  # 0.5179


def agg_by_pipeline(df, feature_set, condition):
    sub = df[(df["feature_set"] == feature_set) & (df["condition"] == condition)]
    rows = []
    for (P, model, hp), grp in sub.groupby(["P", "model", "hyperparams"]):
        mean_epoch = grp["epoch_acc"].mean()
        total_subj = grp["n_subjects_fold"].sum()
        correct = (grp["subject_acc"] * grp["n_subjects_fold"]).sum()
        frac = correct / total_subj if total_subj > 0 else 0.0
        rows.append({"P": P, "model": model, "mean_epoch": mean_epoch,
                     "frac_subjects_correct": frac})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df):,} rows")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
    col_titles = ["PCA", "T-test"]
    row_titles = ["EC (Eyes Closed)", "EO (Eyes Open)"]
    added_labels = set()

    for row_i, cond in enumerate(["EC", "EO"]):
        for col_i, feat in enumerate(["PCA", "ANOVA"]):  # ANOVA = T-test feature selector
            ax = axes[row_i, col_i]
            agg = agg_by_pipeline(df, feat, cond)
            if agg.empty:
                ax.set_title(f"{cond} {feat} -- no data")
                continue

            for model in MODEL_ORDER:
                col = MODEL_COLORS.get(model, "#888")
                for P_val, marker in P_MARKERS.items():
                    pts = agg[(agg["model"] == model) & (agg["P"] == P_val)]
                    if pts.empty:
                        continue
                    lkey = (model, P_val)
                    lbl = f"{model} P={P_val}" if lkey not in added_labels else ""
                    ax.scatter(pts["mean_epoch"], pts["frac_subjects_correct"],
                               color=col, marker=marker, s=55, alpha=0.82,
                               edgecolors="black", linewidth=0.5, zorder=3,
                               label=lbl)
                    if lbl:
                        added_labels.add(lkey)

            # Compute axis limits first (include chance values) so diagonal spans viewport
            x_lo = min(agg["mean_epoch"].min() - 0.02, CHANCE_EPOCH[cond] - 0.03)
            x_hi = agg["mean_epoch"].max() + 0.02
            y_lo = min(agg["frac_subjects_correct"].min() - 0.02, CHANCE_SUBJ - 0.03)
            y_hi = agg["frac_subjects_correct"].max() + 0.02

            # y = x diagonal — spans full axis range
            diag_lo = min(x_lo, y_lo) - 0.01
            diag_hi = max(x_hi, y_hi) + 0.01
            ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", lw=0.9, alpha=0.35)

            # chance lines
            ax.axvline(CHANCE_EPOCH[cond], color="red", lw=1.0, ls=":", alpha=0.55)
            ax.axhline(CHANCE_SUBJ, color="red", lw=1.0, ls=":", alpha=0.55)

            ax.set_xlabel("Epoch accuracy", fontsize=13)
            ax.set_ylabel("Subject accuracy", fontsize=13)
            ax.tick_params(labelsize=13)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

            title = f"{col_titles[col_i]} — {row_titles[row_i]}"
            ax.set_title(title, fontsize=14, fontweight="bold", pad=6)

            # Print stats
            print(f"\n{cond} {feat} P=6:")
            for model in MODEL_ORDER:
                pts6 = agg[(agg["model"]==model) & (agg["P"]==6)]
                if not pts6.empty:
                    print(f"  {model}: epoch={pts6['mean_epoch'].values[0]:.4f}  subj={pts6['frac_subjects_correct'].values[0]:.4f}")

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in labels:
                handles.append(h); labels.append(l)
    from matplotlib.lines import Line2D
    handles += [Line2D([0],[0], color="k", ls="--", lw=1, alpha=0.5, label="y = x  (subject acc = epoch acc)"),
               Line2D([0],[0], color="red", ls=":", lw=1, alpha=0.7, label="Chance")]
    labels  += ["y = x  (subject acc = epoch acc)", "Chance"]
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.16)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=11, framealpha=0.9)
    outfile = OUTDIR / "mdd_trap3_epoch_vs_subject.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved -> {outfile}")


if __name__ == "__main__":
    main()
