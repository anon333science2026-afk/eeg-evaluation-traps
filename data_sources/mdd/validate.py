#!/usr/bin/env python3
"""validate.py — confirm a BIDS-curated MDD tree matches the paper's Table 1.

Checks:
  - 56 subjects have task-eyesclosed (MDD-EC cohort)
  - 59 subjects have task-eyesopen   (MDD-EO cohort)
  - participants.tsv has 32 'mdd' + 29 'control' (full superset)
  - per-cohort group balance:
      MDD-EC: 29 mdd + 27 control
      MDD-EO: 31 mdd + 28 control

Exits 0 if the tree passes, non-zero with a per-check report otherwise.

Usage:
  python data_sources/mdd/validate.py /path/to/mdd_bids
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXPECTED = dict(
    n_ec=56, n_ec_mdd=29, n_ec_ctl=27,
    n_eo=59, n_eo_mdd=31, n_eo_ctl=28,
    n_total_mdd=32, n_total_ctl=29,  # union across both conditions
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bids_root", type=Path, help="Path to the BIDS-formatted MDD tree")
    args = p.parse_args()
    root = args.bids_root.resolve()

    if not root.is_dir():
        print(f"FAIL: {root} is not a directory"); return 1

    results: list[tuple[str, bool, str]] = []

    sub_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    if not sub_dirs:
        print(f"FAIL: no sub-XXX directories under {root}"); return 1

    # Load participants.tsv
    participants_tsv = root / "participants.tsv"
    rows = []
    if participants_tsv.exists():
        with open(participants_tsv) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        results.append(("participants.tsv readable", False, "missing or empty"))
        for name, ok, det in results:
            print(f"  {'✓' if ok else '✗'} {name} — {det}")
        return 1

    group_by_subject = {r["participant_id"]: r.get("group", "") for r in rows}

    # Total cohort counts (union across conditions)
    n_mdd  = sum(1 for g in group_by_subject.values() if g == "mdd")
    n_ctl  = sum(1 for g in group_by_subject.values() if g == "control")
    results.append((
        f"participants.tsv mdd ({n_mdd})",
        n_mdd == EXPECTED["n_total_mdd"],
        f"expected {EXPECTED['n_total_mdd']}, got {n_mdd}",
    ))
    results.append((
        f"participants.tsv control ({n_ctl})",
        n_ctl == EXPECTED["n_total_ctl"],
        f"expected {EXPECTED['n_total_ctl']}, got {n_ctl}",
    ))

    # Per-condition counts via file presence
    for task, n_expected, n_mdd_exp, n_ctl_exp, n_key in (
        ("eyesclosed", EXPECTED["n_ec"], EXPECTED["n_ec_mdd"], EXPECTED["n_ec_ctl"], "EC"),
        ("eyesopen",   EXPECTED["n_eo"], EXPECTED["n_eo_mdd"], EXPECTED["n_eo_ctl"], "EO"),
    ):
        present_subs = []
        for sub_dir in sub_dirs:
            edf = sub_dir / "eeg" / f"{sub_dir.name}_task-{task}_eeg.edf"
            if edf.exists():
                present_subs.append(sub_dir.name)
        n_present = len(present_subs)
        n_mdd_in = sum(1 for s in present_subs if group_by_subject.get(s) == "mdd")
        n_ctl_in = sum(1 for s in present_subs if group_by_subject.get(s) == "control")
        results.append((
            f"task-{task} subjects with EDF ({n_present})",
            n_present == n_expected,
            f"expected {n_expected}, got {n_present}",
        ))
        results.append((
            f"task-{task} mdd ({n_mdd_in})",
            n_mdd_in == n_mdd_exp,
            f"expected {n_mdd_exp}, got {n_mdd_in}",
        ))
        results.append((
            f"task-{task} control ({n_ctl_in})",
            n_ctl_in == n_ctl_exp,
            f"expected {n_ctl_exp}, got {n_ctl_in}",
        ))

    print(f"validate.py — BIDS root: {root}\n")
    n_pass = 0
    for name, ok, detail in results:
        sym = "✓" if ok else "✗"
        print(f"  {sym} {name}" + (f" — {detail}" if not ok else ""))
        n_pass += int(ok)
    print()
    if n_pass == len(results):
        print(f"✅ MDD BIDS CLEAN — all {len(results)} checks PASS")
        return 0
    print(f"❌ MDD BIDS FAIL — {n_pass}/{len(results)} checks pass")
    return 1


if __name__ == "__main__":
    sys.exit(main())
