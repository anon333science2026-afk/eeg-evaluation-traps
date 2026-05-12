#!/usr/bin/env python3
"""
T01 Sanity Check — Table 1 (tab:dataset_summary) — Five-cohort summary.

Paper claims per cohort: (N_disease, N_control, total_epochs,
mean ± SD epochs/subject, ep.chance, subj.chance).

CANONICAL VERIFICATION (post-Round-1 method):
  Read each cohort's processed_subjects parquet (the post-rejection epoch
  matrix that fed ML). Each row = one epoch. Aggregate to compute every
  Table 1 cell directly. No indirect inference.

Tolerance:
  - Counts (N): exact.
  - Total epochs: exact.
  - Mean / SD (paper rounded to integer): ±1.
  - Chance values (paper to 3 dp): ±0.005.

Outputs diff_T01.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent

CANONICAL = {
    "AD":     f"{REPO}/data/canonical/ad/processed_subjects",
    "FTD":    f"{REPO}/data/canonical/ftd/processed_subjects",
    "MDD-EC": f"{REPO}/data/canonical/mdd_ec/processed_subjects",
    "MDD-EO": f"{REPO}/data/canonical/mdd_eo/processed_subjects",
    "SCZ":    f"{REPO}/data/canonical/scz/processed_subjects",
}
DISEASE_GROUP = {"AD": "alz", "FTD": "ftd", "MDD-EC": "mdd", "MDD-EO": "mdd", "SCZ": "scz"}

# Paper claims (post-Round-1 corrections applied)
CLAIMED = {
    "AD":     dict(n_dis=36, n_ctrl=29, total=34044, mean=524, sd=86,  ep_ch=0.547, su_ch=0.554),
    "FTD":    dict(n_dis=23, n_ctrl=29, total=25881, mean=498, sd=95,  ep_ch=0.597, su_ch=0.558),
    "MDD-EC": dict(n_dis=29, n_ctrl=27, total=10978, mean=196, sd=16,  ep_ch=0.503, su_ch=0.518),
    "MDD-EO": dict(n_dis=31, n_ctrl=28, total=11539, mean=196, sd=16,  ep_ch=0.520, su_ch=0.525),
    "SCZ":    dict(n_dis=14, n_ctrl=14, total=18535, mean=662, sd=108, ep_ch=0.535, su_ch=0.500),
}
TOL = dict(n_dis=0, n_ctrl=0, total=0, mean=1, sd=1, ep_ch=0.005, su_ch=0.005)


def compute_cohort(disease):
    df = pd.read_parquet(CANONICAL[disease])
    counts = df.groupby("SubjectID").size()
    n_total = df["SubjectID"].nunique()
    # Disease vs control by Group
    dis_grp = DISEASE_GROUP[disease]
    is_dis = df["Group"] == dis_grp
    dis_subjects = df[is_dis]["SubjectID"].nunique()
    ctl_subjects = df[~is_dis]["SubjectID"].nunique()

    n_eps = len(df)
    mean_e = counts.mean()
    sd_e = counts.std(ddof=1)

    # Epoch chance = majority class fraction
    grp_counts = df["Group"].value_counts()
    ep_ch = grp_counts.max() / n_eps
    # Subject chance
    su_ch = max(dis_subjects, ctl_subjects) / n_total

    return dict(n_dis=dis_subjects, n_ctrl=ctl_subjects, total=n_eps,
                mean=round(float(mean_e)), sd=round(float(sd_e)),
                ep_ch=round(float(ep_ch), 4), su_ch=round(float(su_ch), 4))


def status(claim, comp, tol):
    if comp is None:
        return "MISSING"
    return "PASS" if abs(claim - comp) <= tol else "FAIL"


def main():
    rows = []
    print(f"{'cohort':10s} {'cell':10s} {'claimed':>10s} {'computed':>10s} {'delta':>10s} {'status':>8s}")
    print("-" * 65)
    for disease, claim in CLAIMED.items():
        comp = compute_cohort(disease)
        for cell, claim_val in claim.items():
            comp_val = comp[cell]
            delta = comp_val - claim_val
            stat = status(claim_val, comp_val, TOL[cell])
            rows.append({"cohort": disease, "cell": cell, "claimed": claim_val,
                         "computed": comp_val, "delta": round(delta, 4),
                         "status": stat, "tolerance": TOL[cell]})
            mark = "✓" if stat == "PASS" else "✗"
            print(f"{disease:10s} {cell:10s} {claim_val:>10.4g} {comp_val:>10.4g} {delta:>+10.4g} {stat:>8s} {mark}")

    df = pd.DataFrame(rows)
    out = OUTDIR / "diff_T01.csv"
    df.to_csv(out, index=False)
    n_pass = (df["status"] == "PASS").sum()
    n_total = len(df)
    print()
    print(f"T01 RESULT: {n_pass}/{n_total} cells PASS")
    if n_pass < n_total:
        print(f"  Diff: {out.name}")
        print("\nFailures:")
        print(df[df["status"] != "PASS"].to_string(index=False))


if __name__ == "__main__":
    main()
