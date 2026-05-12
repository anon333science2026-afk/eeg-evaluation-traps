#!/usr/bin/env python3
"""
Appendix N — Per-cohort replication figures.
For each disease (FTD, MDD-EC, MDD-EO, SCZ), produces a single figure with
three sub-panels arranged in a row:
  Panel 1: Trap 1 — violin, subject-overlap vs LPSO P=6
  Panel 2: Trap 2 — boxplot, P=6 vs P=2
  Panel 3: Trap 3 — scatter, epoch vs subject accuracy

Loads the existing per-disease PNG outputs and composes them side-by-side
using matplotlib's image display, preserving exact rendering from source scripts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent

OUTDIR = Path(f"{REPO}/figures/out")

# Disease configs: (label, trap1_png, trap2_png, trap3_png)
DISEASES = [
    {
        "name":  "FTD vs Control",
        "tag":   "FTD",
        "trap1": OUTDIR / "ftd_trap1_CCN.png",
        "trap2": OUTDIR / "ftd_trap2_CCN.png",
        "trap3": OUTDIR / "ftd_trap3_epoch_vs_subject_P6_P2.png",
    },
    {
        "name":  "MDD vs Control — Eyes Closed (EC)",
        "tag":   "MDD_EC",
        "trap1": OUTDIR / "mdd_trap1_CCN.png",
        "trap2": OUTDIR / "mdd_trap2_CCN.png",
        "trap3": OUTDIR / "mdd_trap3_epoch_vs_subject.png",
        "trap1_row": 0,   # 2×2 figure: use top row (EC)
    },
    {
        "name":  "MDD vs Control — Eyes Open (EO)",
        "tag":   "MDD_EO",
        "trap1": OUTDIR / "mdd_trap1_CCN.png",
        "trap2": OUTDIR / "mdd_trap2_CCN.png",
        "trap3": OUTDIR / "mdd_trap3_epoch_vs_subject.png",
        "trap1_row": 1,   # 2×2 figure: use bottom row (EO)
    },
    {
        "name":  "SCZ vs Control",
        "tag":   "SCZ",
        "trap1": OUTDIR / "scz_trap1_CCN.png",
        "trap2": OUTDIR / "scz_trap2_CCN.png",
        "trap3": OUTDIR / "scz_trap3_epoch_vs_subject_P6_P2.png",
    },
]


def load_image(path: Path):
    if not path.exists():
        print(f"  WARNING: {path.name} not found")
        return None
    return mpimg.imread(str(path))


def crop_half_rows(img, row: int):
    """Crop top or bottom half from a stacked-row figure (row=0: top, row=1: bottom)."""
    h = img.shape[0]
    mid = h // 2
    if row == 0:
        return img[:mid, :, :]
    else:
        return img[mid:, :, :]


def make_panel_figure(disease: dict):
    name = disease["name"]
    tag  = disease["tag"]
    t1   = load_image(disease["trap1"])
    t2   = load_image(disease["trap2"])
    t3   = load_image(disease["trap3"])

    # For MDD, the Trap 1 and Trap 2 figures are 2-row — crop to the relevant row
    if "trap1_row" in disease and t1 is not None:
        t1 = crop_half_rows(t1, disease["trap1_row"])
    if "trap1_row" in disease and t2 is not None:
        t2 = crop_half_rows(t2, disease["trap1_row"])

    # Pyramidal layout: Trap 1 full-width top, Trap 2 bottom-left, Trap 3 bottom-right
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.06,
                          height_ratios=[1, 1])

    ax_top  = fig.add_subplot(gs[0, :])   # Trap 1 spans both columns
    ax_bl   = fig.add_subplot(gs[1, 0])   # Trap 2 bottom-left
    ax_br   = fig.add_subplot(gs[1, 1])   # Trap 3 bottom-right

    panels = [
        (ax_top, t1, "Trap 1: Subject-overlap vs Disjoint"),
        (ax_bl,  t2, "Trap 2: P=6 vs P=2 IQR Inflation"),
        (ax_br,  t3, "Trap 3: Epoch vs Subject Accuracy"),
    ]
    for ax, img, subtitle in panels:
        ax.axis("off")
        if img is not None:
            ax.imshow(img, interpolation="lanczos")
            ax.set_aspect("equal")
        ax.set_title(subtitle, fontsize=12, fontweight="bold", pad=4)

    fig.suptitle(name, fontsize=15, fontweight="bold", y=1.01)

    outfile = OUTDIR / f"appendix_N_{tag}.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {outfile}")


for disease in DISEASES:
    print(f"\n{'='*60}")
    print(f"Compositing: {disease['name']}")
    make_panel_figure(disease)

print("\nAll Appendix N figures done.")
