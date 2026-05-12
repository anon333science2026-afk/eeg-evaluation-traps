#!/usr/bin/env python3
"""
T13 Sanity Check — Table 13 (tab:top_features) — Top-5 T-test features per disease.

Paper claim: top-5 features per disease by |Cohen's d|, with arrow showing direction
(↑ = higher in disease, ↓ = lower in disease) and d value to 2 dp.

Verification strategy:
  • Read each disease's canonical parquet directly (same sources as T01).
  • Extract 95-feature matrix per epoch.
  • Compute Cohen's d per feature (pooled SD).
  • Map feature_idx → (channel, band) using each disease's existing
    top_anova_features_<DIS>.csv index map.
  • Take top-5 by |d|, compare label, direction, and d.

Tolerance: d to ±0.005 (paper to 2 dp).

Outputs diff_T13.csv + table_T13_recomputed.csv.
"""

import pandas as pd, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent

CANONICAL = {
    "AD":     f"{REPO}/processed_data/canonical/ad/processed_subjects",
    "FTD":    f"{REPO}/processed_data/canonical/ftd/processed_subjects",
    "MDD-EC": f"{REPO}/processed_data/canonical/mdd_ec/processed_subjects",
    "MDD-EO": f"{REPO}/processed_data/canonical/mdd_eo/processed_subjects",
    "SCZ":    f"{REPO}/processed_data/canonical/scz/processed_subjects",
}
INDEX_MAP = {
    "AD":     f"{REPO}/processed_data/per_subject_summaries/top_anova_features_AD.csv",
    "FTD":    f"{REPO}/processed_data/per_subject_summaries/top_anova_features_FTD.csv",
    "MDD-EC": f"{REPO}/processed_data/per_subject_summaries/top_anova_features_MDD_EC.csv",
    "MDD-EO": f"{REPO}/processed_data/per_subject_summaries/top_anova_features_MDD_EO.csv",
    "SCZ":    f"{REPO}/processed_data/per_subject_summaries/top_anova_features_SCZ.csv",
}
DISEASE_GROUP = {"AD":"alz", "FTD":"ftd", "MDD-EC":"mdd", "MDD-EO":"mdd", "SCZ":"scz"}

# ── Paper claims (Table 13) — (label, arrow, |d|) for ranks 1-5 per disease ──
# arrow ↓ = lower in disease, ↑ = higher in disease
PAPER = {
    "AD":     [("O2_Alpha","↓",0.82), ("T5_Alpha","↓",0.74), ("O1_Alpha","↓",0.72),
               ("T6_Alpha","↓",0.62), ("O2_Delta","↑",0.62)],
    "FTD":    [("T5_Alpha","↓",0.56), ("O2_Alpha","↓",0.52), ("P3_Alpha","↓",0.48),
               ("O1_Alpha","↓",0.47), ("O2_Delta","↑",0.43)],
    "MDD-EC": [("C3_Beta","↑",1.22), ("Pz_Beta","↑",1.16), ("P4_Beta","↑",1.15),
               ("T6_Beta","↑",1.13), ("P3_Beta","↑",1.10)],
    "MDD-EO": [("P4_Beta","↑",1.12), ("Pz_Beta","↑",1.12), ("O1_Beta","↑",1.08),
               ("P3_Beta","↑",1.07), ("T6_Beta","↑",1.05)],
    "SCZ":    [("T3_Alpha","↑",0.32), ("Fp2_Beta","↑",0.30), ("Fp2_Alpha","↑",0.29),
               ("Fp2_Delta","↓",0.28), ("P4_Theta","↓",0.27)],
}

TOL_D = 0.005


def cohen_d(x_dis, x_ctl):
    n1, n2 = len(x_dis), len(x_ctl)
    s1, s2 = x_dis.std(ddof=1), x_ctl.std(ddof=1)
    s_pool = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (x_dis.mean() - x_ctl.mean()) / s_pool


def compute_top_features(parquet_path, idx_map_csv, disease_group):
    df = pd.read_parquet(parquet_path)
    feats = np.array([np.array(v["values"]) for v in df["features"]])
    is_dis = (df["Group"] == disease_group).values
    rows = []
    for i in range(feats.shape[1]):
        x = feats[:, i]
        d = cohen_d(x[is_dis], x[~is_dis])
        rows.append({"feature_idx": i, "cohen_d": d, "abs_d": abs(d)})
    full = pd.DataFrame(rows)
    # attach label
    label_map = pd.read_csv(idx_map_csv)[["feature_idx","label"]]
    full = full.merge(label_map, on="feature_idx", how="left")
    full = full.sort_values("abs_d", ascending=False).reset_index(drop=True)
    full["rank"] = full.index + 1
    full["arrow"] = np.where(full["cohen_d"] > 0, "↑", "↓")
    return full


def main():
    diff_rows = []
    all_recomputed = []
    print(f"{'disease':8s} {'rank':>4s} | {'paper':<22s} | {'computed':<22s} | label match | d match")
    print("-" * 100)
    for disease, paper_rows in PAPER.items():
        full = compute_top_features(CANONICAL[disease], INDEX_MAP[disease], DISEASE_GROUP[disease])
        top5 = full.head(5).copy()
        top5.insert(0, "disease", disease)
        all_recomputed.append(top5)
        for rank, (paper_label, paper_arrow, paper_d) in enumerate(paper_rows, 1):
            comp = full.iloc[rank-1]
            comp_label = comp["label"]
            comp_arrow = comp["arrow"]
            comp_d = comp["abs_d"]
            label_match = comp_label == paper_label
            arrow_match = comp_arrow == paper_arrow
            d_match = abs(comp_d - paper_d) <= TOL_D
            status = "PASS" if (label_match and arrow_match and d_match) else "FAIL"
            note = []
            if not label_match: note.append(f"label {comp_label} vs {paper_label}")
            if not arrow_match: note.append(f"arrow {comp_arrow} vs {paper_arrow}")
            if not d_match:     note.append(f"d {comp_d:.3f} vs {paper_d}")
            diff_rows.append({"disease": disease, "rank": rank,
                              "paper_label": paper_label, "paper_arrow": paper_arrow, "paper_d": paper_d,
                              "comp_label": comp_label, "comp_arrow": comp_arrow, "comp_d": round(comp_d,4),
                              "status": status, "notes": "; ".join(note)})
            mark = "✓" if status == "PASS" else "✗"
            ps = f"{paper_label}{paper_arrow} ({paper_d:.2f})"
            cs = f"{comp_label}{comp_arrow} ({comp_d:.2f})"
            print(f"{disease:8s} {rank:>4d} | {ps:<22s} | {cs:<22s} | {('✓' if label_match else '✗'):>11s} | {('✓' if d_match else '✗'):>7s} {mark}")

    diff = pd.DataFrame(diff_rows)
    out_diff = OUTDIR / "diff_T13.csv"
    diff.to_csv(out_diff, index=False)
    out_table = OUTDIR / "table_T13_recomputed.csv"
    pd.concat(all_recomputed, ignore_index=True)[
        ["disease","rank","label","arrow","cohen_d","abs_d","feature_idx"]
    ].to_csv(out_table, index=False)

    n_pass = (diff["status"] == "PASS").sum()
    n_total = len(diff)
    print()
    print(f"T13 RESULT: {n_pass}/{n_total} cells PASS")
    print(f"  Diff CSV:    {out_diff.name}")
    print(f"  New table:   {out_table.name}")
    if n_pass < n_total:
        print("\nFailing rows:")
        print(diff[diff["status"]=="FAIL"][["disease","rank","paper_label","comp_label","paper_d","comp_d","notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
