#!/usr/bin/env python3
"""
figA.1.7.2 — mismatch variants, both cases from Random-50.
  Row 1: SVM / T-test  — epoch overestimates subject (Δ = +20.9 pp)
  Row 2: KNN k=7 / T-test — subject overestimates epoch (Δ = +32.8 pp)
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

BASE   = Path(f"{REPO}/data/ad_extras")
PREDS  = BASE / "per_subject_fold_predictions.csv"
MERGED = BASE / "fold_epoch_vs_subject_merged.csv"
OUTDIR = Path(f"{REPO}/figures/out")

CLASS_COL = {0: "#27ae60", 1: "#2980b9"}   # green=Control, blue=AD

merged   = pd.read_csv(MERGED)
preds_df = pd.read_csv(PREDS)
merged["mismatch"] = (merged["epoch_acc"] - merged["subject_acc"]) * 100
m6 = merged[merged["P"] == 6]

# Case 1: SVM / T-test / Random-50 — largest epoch > subject gap in Random-50
case1 = m6[
    (m6["pipeline"]  == "FTest") &
    (m6["strategy"]  == "Random-50") &
    (m6["model"]     == "SVM") &
    (m6["mismatch"]  > 0)
].nlargest(1, "mismatch").iloc[0]

# Case 2: KNN k=7 / T-test / Random-50 — largest subject > epoch gap
case2 = m6[
    (m6["pipeline"]  == "FTest") &
    (m6["strategy"]  == "Random-50") &
    (m6["model"]     == "KNN") &
    (m6["hyperparams"].str.contains('"n_neighbors": 7')) &
    (m6["mismatch"]  < 0)
].nsmallest(1, "mismatch").iloc[0]


def get_fold_preds(case):
    return preds_df[
        (preds_df["experiment"] == case["experiment"]) &
        (preds_df["pipeline"]   == case["pipeline"])   &
        (preds_df["strategy"]   == case["strategy"])   &
        (preds_df["P"]          == case["P"])           &
        (preds_df["fold_id"]    == case["fold_id"])     &
        (preds_df["model"]      == case["model"])       &
        (preds_df["hyperparams"]== case["hyperparams"])
    ].copy()


def prep_fold(fp):
    subjects      = fp["SubjectID"].values
    true_labels   = fp["true_subject_label"].values.astype(int)
    correct       = fp["subject_correct"].values.astype(bool)
    n_epochs      = fp["n_epochs"].values.astype(float)
    ad_ratio      = fp["ad_ratio"].values
    correct_votes = (ad_ratio * n_epochs).round().astype(int)
    other_votes   = (n_epochs - correct_votes).astype(int)
    order = np.lexsort((n_epochs * -1, true_labels * -1))
    return (subjects[order], true_labels[order], correct[order],
            n_epochs[order], correct_votes[order], other_votes[order])


def draw_bars(ax, subjects, true_labels, correct, n_epochs,
              correct_votes, other_votes):
    x = np.arange(len(subjects))
    for i, (cv, ov, tl) in enumerate(zip(correct_votes, other_votes, true_labels)):
        col = CLASS_COL[tl]
        ax.bar(i, cv, color=col, edgecolor="black", linewidth=0.7, alpha=0.85, width=0.6)
        ax.bar(i, ov, bottom=cv, color=col, edgecolor="black", linewidth=0.7, alpha=0.25, width=0.6)

    for i, (cv, ne, c) in enumerate(zip(correct_votes, n_epochs, correct)):
        sym_col = "#27ae60" if c else "#e74c3c"
        sym     = r"$\checkmark$" if c else r"$\times$"
        ax.text(i, ne + n_epochs.max() * 0.03, sym, ha="center", va="bottom",
                fontsize=26, fontweight="bold", color=sym_col)
        ax.text(i, ne + n_epochs.max() * 0.01, f"{cv}/{int(ne)}",
                ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Sub-{s}" for s in subjects], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Number of epoch votes", fontsize=12)
    ax.set_ylim(0, n_epochs.max() * 1.28)
    ax.legend(handles=[
        mpatches.Patch(facecolor=CLASS_COL[1], label="Alzheimer's subject"),
        mpatches.Patch(facecolor=CLASS_COL[0], label="Control subject"),
    ], fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_accuracy(ax, epoch_acc_pct, subject_acc_pct, delta):
    vals = [epoch_acc_pct, subject_acc_pct]
    bars = ax.bar([0, 1], vals, color=["#4a9ede", "#e07b39"],
                  edgecolor="black", linewidth=0.9, alpha=0.88, width=0.55)
    hi, lo = max(vals), min(vals)
    hi_x = 0 if vals[0] > vals[1] else 1
    lo_x = 1 - hi_x
    ax.fill_between([lo_x - 0.275, lo_x + 0.275], lo, hi,
                    color="#ffeb3b", alpha=0.55, zorder=0,
                    label=f"Gap = {abs(delta):.1f} pp")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Epoch accuracy\n(weighted by epoch count)",
                        "Subject-label accuracy\n(1 vote per subject)"], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11.5),
                         gridspec_kw={"hspace": 0.32, "wspace": 0.28})

# Row 1 — SVM / T-test (epoch overestimates subject)
fp1 = get_fold_preds(case1)
s, tl, c, ne, cv, ov = prep_fold(fp1)
epoch1   = case1["epoch_acc"]   * 100
subject1 = case1["subject_acc"] * 100
delta1   = epoch1 - subject1
draw_bars(axes[0, 0], s, tl, c, ne, cv, ov)
draw_accuracy(axes[0, 1], epoch1, subject1, delta1)

# Row 2 — KNN k=7 / T-test (subject overestimates epoch)
fp2 = get_fold_preds(case2)
s2, tl2, c2, ne2, cv2, ov2 = prep_fold(fp2)
epoch2   = case2["epoch_acc"]   * 100
subject2 = case2["subject_acc"] * 100
delta2   = subject2 - epoch2
draw_bars(axes[1, 0], s2, tl2, c2, ne2, cv2, ov2)
draw_accuracy(axes[1, 1], epoch2, subject2, -(delta2))

plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Horizontal row titles above each row
for row, label in enumerate([
    "Epoch Accuracy Overestimates Subject Accuracy",
    "Subject Accuracy Overestimates Epoch Accuracy",
]):
    pos_l = axes[row, 0].get_position()
    pos_r = axes[row, 1].get_position()
    x_center = (pos_l.x0 + pos_r.x1) / 2
    y_top    = pos_l.y0 + pos_l.height + 0.012
    fig.text(x_center, y_top, label,
             fontsize=15, fontweight="bold", rotation=0,
             ha="center", va="bottom", color="#222222")

out = OUTDIR / "figA.1.7.2_mismatch_SVM_KNN.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Case 1 (SVM,  epoch>subj): epoch={epoch1:.1f}%  subj={subject1:.1f}%  Δ={delta1:+.1f} pp")
print(f"Case 2 (KNN,  subj>epoch): epoch={epoch2:.1f}%  subj={subject2:.1f}%  Δ=+{delta2:.1f} pp")
print(f"\nCase 1 fold: {case1['fold_id']}")
print(f"Case 2 fold: {case2['fold_id']}")
