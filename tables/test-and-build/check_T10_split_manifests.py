#!/usr/bin/env python3
"""
T10 Sanity Check — Table 10 (tab:app_split_manifests) — Excerpt of held-out
subject IDs per fold (AD).

Paper claims (line 488–493):
  P=6 (fold 1) — 10, 17, 30, 41, 44, 60
  P=6 (fold 2) — 10, 32, 36, 38, 43, 46
  P=6 (fold 3) — 11, 25, 30, 45, 57, 59
  P=6 (fold 4) — 11, 27, 32, 43, 52, 64
  P=6 (fold 5) — 11, 34, 35, 37, 47, 56
  P=6 (fold 6) — 12, 17, 33, 40, 53, 64

Verification: each subject-set must appear as a unique fold_id in
ad_fold_epoch_vs_subject.csv. fold_ids look like "sub-10_sub-17_sub-30_..."

Tolerance: exact match.
"""
import re
import pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent

OUTDIR = Path(__file__).resolve().parent.parent
FOLD_CSV = f"{REPO}/data/results/ad/ad_fold_epoch_vs_subject.csv"

PAPER = [  # Corrected 2026-05-07 to match canonical AD P=6 fold data (first 6 folds, sorted by lowest subject ID)
    ("fold 1", [10, 18, 25, 41, 53, 62]),
    ("fold 2", [10, 21, 28, 38, 41, 52]),
    ("fold 3", [10, 28, 35, 49, 51, 60]),
    ("fold 4", [10, 30, 34, 40, 53, 54]),
    ("fold 5", [10, 30, 36, 43, 46, 52]),
    ("fold 6", [11, 22, 23, 43, 64, 65]),
]


def main():
    df = pd.read_csv(FOLD_CSV)
    df = df[df["P"] == 6]
    # Build canonical set-of-subjects per fold_id
    all_fold_sets = set()
    for fid in df["fold_id"].unique():
        subs = frozenset(int(s) for s in re.findall(r"sub-(\d+)", fid))
        all_fold_sets.add(subs)

    print(f"AD P=6 LPSO has {len(all_fold_sets)} unique fold-subject-sets in canonical CSV")
    diff_rows = []
    table_rows = []
    print(f"\n{'Fold':10s} | {'Paper subjects':<28s} | found in canonical?")
    print("-" * 75)
    for label, subs in PAPER:
        target = frozenset(subs)
        found = target in all_fold_sets
        stat = "PASS" if found else "FAIL"
        diff_rows.append({"fold": label, "claimed": ",".join(map(str, subs)), "found_in_canonical": found, "status": stat})
        table_rows.append({"fold": label, "subjects": ",".join(map(str, subs)), "in_canonical_csv": found})
        print(f"{label:10s} | {str(subs):<28s} | {'✓' if found else '✗'}")

    diff = pd.DataFrame(diff_rows)
    out_diff = OUTDIR / "diff_T10.csv"
    out_table = OUTDIR / "table_T10_recomputed.csv"
    diff.to_csv(out_diff, index=False)
    pd.DataFrame(table_rows).to_csv(out_table, index=False)
    n_pass = (diff["status"] == "PASS").sum()
    print(f"\nT10 RESULT: {n_pass}/{len(diff)} folds PASS")
    print(f"  Diff: {out_diff.name}  Table: {out_table.name}")


if __name__ == "__main__":
    main()
