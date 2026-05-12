#!/usr/bin/env python3
"""validate.py — confirm a BIDS-curated SCZ tree matches the paper's Table 1.

Checks:
  - 28 subject directories (sub-001 .. sub-028)
  - participants.tsv has 14 'schizophrenia' + 14 'control'
  - each subject has an EEG .edf file at sub-XXX/eeg/sub-XXX_task-<task>_eeg.edf

Exits 0 if the tree passes, non-zero with a per-check report otherwise.

Usage:
  python data_sources/scz/validate.py /path/to/scz_bids
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXPECTED = dict(n_subjects=28, n_schizophrenia=14, n_control=14, task="rest")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bids_root", type=Path, help="Path to the BIDS-formatted SCZ tree")
    args = p.parse_args()
    root = args.bids_root.resolve()

    if not root.is_dir():
        print(f"FAIL: {root} is not a directory"); return 1

    results: list[tuple[str, bool, str]] = []

    sub_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    results.append((
        f"subject dirs ({len(sub_dirs)})",
        len(sub_dirs) == EXPECTED["n_subjects"],
        f"expected {EXPECTED['n_subjects']}, got {len(sub_dirs)}",
    ))

    participants_tsv = root / "participants.tsv"
    if not participants_tsv.exists():
        results.append(("participants.tsv exists", False, "missing"))
    else:
        with open(participants_tsv) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        groups = [r.get("group", "") for r in rows]
        n_scz = sum(1 for g in groups if g == "schizophrenia")
        n_ctl = sum(1 for g in groups if g == "control")
        results.append((
            f"participants.tsv schizophrenia ({n_scz})",
            n_scz == EXPECTED["n_schizophrenia"],
            f"expected {EXPECTED['n_schizophrenia']}, got {n_scz}",
        ))
        results.append((
            f"participants.tsv control ({n_ctl})",
            n_ctl == EXPECTED["n_control"],
            f"expected {EXPECTED['n_control']}, got {n_ctl}",
        ))

    missing_edfs: list[str] = []
    for sub_dir in sub_dirs:
        edf = sub_dir / "eeg" / f"{sub_dir.name}_task-{EXPECTED['task']}_eeg.edf"
        if not edf.exists():
            missing_edfs.append(str(edf.relative_to(root)))
    results.append((
        f"per-subject EDFs present ({len(sub_dirs) - len(missing_edfs)}/{len(sub_dirs)})",
        not missing_edfs,
        f"missing: {missing_edfs[:5]}" if missing_edfs else "all present",
    ))

    print(f"validate.py — BIDS root: {root}\n")
    n_pass = 0
    for name, ok, detail in results:
        sym = "✓" if ok else "✗"
        print(f"  {sym} {name}" + (f" — {detail}" if not ok else ""))
        n_pass += int(ok)
    print()
    if n_pass == len(results):
        print(f"✅ SCZ BIDS CLEAN — all {len(results)} checks PASS")
        return 0
    print(f"❌ SCZ BIDS FAIL — {n_pass}/{len(results)} checks pass")
    return 1


if __name__ == "__main__":
    sys.exit(main())
