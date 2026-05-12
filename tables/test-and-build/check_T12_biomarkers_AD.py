#!/usr/bin/env python3
"""
T12 Sanity Check — Table 12 (tab:biomarkers) — AD top-10 EEG biomarkers.

Paper caption: "Top 10 EEG biomarkers for AD vs control classification
(Cohen's |d|; 800µV rejection threshold, 34,044 epochs)."

Verification strategy:
  • Read the canonical AD parquet (34,044 epochs) directly.
  • Extract the 95-dim feature matrix per epoch (Spark dense-vector "values" field).
  • For each feature, compute Cohen's d using the standard pooled-SD definition:
        d = (mean_disease - mean_control) / s_pooled
        s_pooled = sqrt(((n1-1)·s1² + (n2-1)·s2²) / (n1 + n2 - 2))
  • Map feature_idx → (channel, band) using the existing
    top_anova_features_AD.csv mapping (the index→label map is the only
    thing we still trust in that CSV; the d values themselves are stale).
  • Sort features by |d| descending, take top 10.
  • Compare each cell of paper Table 12 against recomputed values.

Tolerance: ±0.005 for Cohen's d (paper to 3 dp), ±0.01 for ratio,
±0.001 for raw means (paper to 4 dp).

Outputs:
  diff_T12.csv — per-cell pass/fail
  table_T12_recomputed.csv — full top-10 in paper format, ready to copy in.
"""

import pandas as pd, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
PARQUET = f"{REPO}/processed_data/canonical/ad/processed_subjects"
INDEX_MAP_CSV = f"{REPO}/processed_data/per_subject_summaries/top_anova_features_AD.csv"

# ── Paper claims (Table 12, rows in published rank order) ────────────────────
PAPER = [
    # (label, mean_AD, mean_control, ratio, cohen_d)
    ("O2_Alpha", 0.0400, 0.1064, 2.66, 0.822),
    ("T5_Alpha", 0.0364, 0.0916, 2.51, 0.739),
    ("O1_Alpha", 0.0389, 0.0951, 2.45, 0.725),
    ("T6_Alpha", 0.0379, 0.0772, 2.04, 0.617),
    ("O2_Delta", 0.8888, 0.8260, 1.08, 0.616),
    ("P3_Alpha", 0.0278, 0.0538, 1.94, 0.547),
    ("O1_Delta", 0.8898, 0.8378, 1.06, 0.521),
    ("T5_Delta", 0.8896, 0.8389, 1.06, 0.512),
    ("Pz_Alpha", 0.0259, 0.0480, 1.86, 0.508),
    ("P4_Alpha", 0.0285, 0.0470, 1.65, 0.434),
]

TOL = dict(mean=0.001, ratio=0.01, d=0.005)


def cohen_d(x_dis, x_ctl):
    n1, n2 = len(x_dis), len(x_ctl)
    s1, s2 = x_dis.std(ddof=1), x_ctl.std(ddof=1)
    s_pool = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (x_dis.mean() - x_ctl.mean()) / s_pool


def main():
    df = pd.read_parquet(PARQUET)
    print(f"Loaded canonical AD parquet: {len(df):,} epochs, {df['SubjectID'].nunique()} subjects")

    # Feature index → (channel, band) name mapping from the existing CSV
    idx_map = pd.read_csv(INDEX_MAP_CSV)[["feature_idx", "label"]]
    idx_to_label = dict(zip(idx_map["feature_idx"], idx_map["label"]))

    # Extract feature matrix (n_epochs × 95)
    feats = np.array([np.array(v["values"]) for v in df["features"]])
    n_features = feats.shape[1]

    is_disease = (df["Group"] == "alz").values
    n_dis  = int(is_disease.sum())
    n_ctl  = int((~is_disease).sum())
    print(f"Disease epochs: {n_dis}, Control epochs: {n_ctl}")

    # Compute Cohen's d for each feature
    rows = []
    for i in range(n_features):
        x = feats[:, i]
        x_dis = x[is_disease]
        x_ctl = x[~is_disease]
        d = cohen_d(x_dis, x_ctl)
        m_dis = x_dis.mean()
        m_ctl = x_ctl.mean()
        ratio = max(m_dis, m_ctl) / min(m_dis, m_ctl) if min(m_dis, m_ctl) > 0 else float("nan")
        rows.append({
            "feature_idx": i,
            "label":   idx_to_label.get(i, f"f{i}"),
            "mean_disease": m_dis,
            "mean_control": m_ctl,
            "ratio":   ratio,
            "cohen_d": d,
            "abs_d":   abs(d),
        })
    full = pd.DataFrame(rows).sort_values("abs_d", ascending=False).reset_index(drop=True)
    full["rank"] = full.index + 1

    top10 = full.head(10).copy()
    out_table = OUTDIR / "table_T12_recomputed.csv"
    top10.to_csv(out_table, index=False)
    print(f"\nRecomputed top-10 (saved to {out_table.name}):")
    print(top10[["rank","label","mean_disease","mean_control","ratio","cohen_d","abs_d"]].to_string(index=False))

    # ── Diff vs paper claims ──────────────────────────────────────────────────
    diff_rows = []
    print(f"\n{'rank':>4s} {'label':<10s} | {'cell':<6s} {'paper':>8s} {'computed':>10s} {'delta':>10s} {'status':>8s}")
    print("-" * 78)
    for rank, (label, m_dis_p, m_ctl_p, ratio_p, d_p) in enumerate(PAPER, start=1):
        # find by label in computed
        match = full[full["label"] == label]
        if match.empty:
            print(f"  ⚠ paper feature '{label}' (rank {rank}) not in computed top — listing absolute rank below")
            match = full[full["label"] == label]
        # report regardless of computed rank
        comp = full[full["label"] == label].iloc[0] if not full[full["label"]==label].empty else None
        comp_rank = int(comp["rank"]) if comp is not None else None
        for cell, paper_v, comp_v, tol in [
            ("rank",   rank,    comp_rank,                     0),
            ("mean_d", m_dis_p, comp["mean_disease"] if comp is not None else None, TOL["mean"]),
            ("mean_c", m_ctl_p, comp["mean_control"] if comp is not None else None, TOL["mean"]),
            ("ratio",  ratio_p, comp["ratio"]        if comp is not None else None, TOL["ratio"]),
            ("d",      d_p,     comp["abs_d"]        if comp is not None else None, TOL["d"]),
        ]:
            if comp_v is None:
                status = "MISSING"; delta = None
            else:
                delta = comp_v - paper_v
                status = "PASS" if abs(delta) <= tol else "FAIL"
            diff_rows.append({"rank":rank, "label":label, "cell":cell,
                              "claimed":paper_v, "computed":comp_v, "delta":delta,
                              "status":status, "tolerance":tol})
            d_str = f"{delta:+.4f}" if delta is not None else "  --  "
            print(f"  {rank:>2d}  {label:<10s} | {cell:<6s} {paper_v:>8.4f} {('%.4f' % comp_v) if comp_v is not None else '--':>10s} {d_str:>10s} {status:>8s}")

    diff_df = pd.DataFrame(diff_rows)
    diff_csv = OUTDIR / "diff_T12.csv"
    diff_df.to_csv(diff_csv, index=False)
    n_pass = (diff_df["status"] == "PASS").sum()
    n_total = len(diff_df)
    print()
    print(f"T12 RESULT: {n_pass}/{n_total} cells PASS")
    print(f"  Diff CSV:           {diff_csv.name}")
    print(f"  Recomputed table:   {out_table.name}")


if __name__ == "__main__":
    main()
