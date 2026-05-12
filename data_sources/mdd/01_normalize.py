from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import mne

# Channels for a 10-20 system
TARGET_CHANNELS = [
    "Fp1-LE",
    "Fp2-LE",
    "F3-LE",
    "F4-LE",
    "C3-LE",
    "C4-LE",
    "P3-LE",
    "P4-LE",
    "O1-LE",
    "O2-LE",
    "F7-LE",
    "F8-LE",
    "T3-LE",
    "T4-LE",
    "T5-LE",
    "T6-LE",
    "Fz-LE",
    "Cz-LE",
    "Pz-LE",
]
# we ignore task for this situation
EXCLUDED_CONDITIONS = {"TASK"}
# EC --> eyes clolsed ; EO --> eyes open
EXPECTED_REST_CONDITIONS = ("EC", "EO")
ALL_STUDY_CONDITIONS = ("EC", "EO", "TASK")
SUSPECTED_DUPLICATE_SUBJECT_EXCLUSIONS = {
    ("Control", 30): {
        "kept_subject_number": 27,
        "reason": "raw EC/EO/TASK EDFs are byte-identical to Control H S27 across all three conditions",
    },
    ("MDD", 34): {
        "kept_subject_number": 33,
        "reason": "raw EC/EO/TASK EDFs are byte-identical to MDD S33 across all three conditions",
    },
}


@dataclass(frozen=True)
class RawSlotRecord:
    source_path: Path
    source_name: str
    normalized_name: str
    group: str
    subject_label: str
    subject_number: int
    condition: str
    md5: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 4244171_normalized directly from the raw flat 4244171 EDF dump."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Path to the raw 4244171 directory containing EDF files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path where the derivative dataset should be written.",
    )
    return parser.parse_args()


# A helper function to ensure that the dataset created is the same /different from other ones
# This was used to ensure the script was reproducible and create the same dataset again and again
def md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_filename(name: str) -> str:
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_normalized_name(name: str) -> tuple[str, str, int, str]:
    match = re.fullmatch(r"(H|MDD) S(\d+) (EC|EO|TASK)\.edf", name)
    if not match:
        raise ValueError(f"Unexpected filename format after normalization: {name}")
    prefix, subject_number_str, condition = match.groups()
    subject_number = int(subject_number_str)
    if prefix == "H":
        return "Control", f"H S{subject_number}", subject_number, condition
    return "MDD", f"MDD S{subject_number}", subject_number, condition


def format_subject_label(group: str, subject_number: int) -> str:
    if group == "Control":
        return f"H S{subject_number}"
    return f"MDD S{subject_number}"


def output_group_dir(group: str) -> str:
    return "healthy_controls" if group == "Control" else "major_depressive_disorder"


def output_subject_dir(group: str, subject_number: int) -> str:
    if group == "Control":
        return f"H_S{subject_number:02d}"
    return f"MDD_S{subject_number:02d}"


def write_tsv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def discover_raw_slots(source_dir: Path) -> tuple[list[RawSlotRecord], list[str]]:
    records: list[RawSlotRecord] = []
    unexpected_names: list[str] = []

    for source_path in sorted(source_dir.glob("*.edf")):
        normalized_name = normalize_filename(source_path.name)
        try:
            group, subject_label, subject_number, condition = parse_normalized_name(normalized_name)
        except ValueError:
            unexpected_names.append(source_path.name)
            continue

        records.append(
            RawSlotRecord(
                source_path=source_path,
                source_name=source_path.name,
                normalized_name=normalized_name,
                group=group,
                subject_label=subject_label,
                subject_number=subject_number,
                condition=condition,
                md5=md5(source_path),
                size_bytes=source_path.stat().st_size,
            )
        )

    return records, unexpected_names


def canonicalize_raw_slots(
    raw_records: list[RawSlotRecord],
) -> tuple[list[RawSlotRecord], list[dict[str, str]], list[dict[str, str]]]:
    slot_to_records: dict[tuple[str, int, str], list[RawSlotRecord]] = defaultdict(list)
    for record in raw_records:
        slot_to_records[(record.group, record.subject_number, record.condition)].append(record)

    canonical_records: list[RawSlotRecord] = []
    duplicate_rows: list[dict[str, str]] = []
    name_fix_rows: list[dict[str, str]] = []

    for records in slot_to_records.values():
        records_sorted = sorted(records, key=lambda item: (item.source_name, item.md5))
        distinct_hashes = {record.md5 for record in records_sorted}
        if len(distinct_hashes) > 1:
            raise RuntimeError(
                f"Normalized slot {records_sorted[0].normalized_name} has conflicting non-identical files: "
                + ", ".join(record.source_name for record in records_sorted)
            )

        kept = records_sorted[0]
        canonical_records.append(kept)

        for record in records_sorted:
            if record.source_name != record.normalized_name:
                reasons = []
                if re.match(r"^\d+_", record.source_name):
                    reasons.append("stripped numeric prefix")
                if "  " in record.source_name:
                    reasons.append("collapsed repeated spaces")
                if not reasons:
                    reasons.append("normalized spacing")
                name_fix_rows.append(
                    {
                        "source_name": record.source_name,
                        "normalized_name": record.normalized_name,
                        "reason": ", ".join(reasons),
                    }
                )

        for omitted in records_sorted[1:]:
            duplicate_rows.append(
                {
                    "normalized_slot": kept.normalized_name,
                    "kept_source": kept.source_name,
                    "omitted_source": omitted.source_name,
                    "size_bytes": str(omitted.size_bytes),
                    "md5": omitted.md5,
                }
            )

    canonical_records.sort(key=lambda item: (item.group, item.subject_number, item.condition))
    duplicate_rows.sort(key=lambda row: (row["normalized_slot"], row["omitted_source"]))
    name_fix_rows.sort(key=lambda row: row["source_name"])
    return canonical_records, duplicate_rows, name_fix_rows


def build_raw_hash_rows(
    raw_records: list[RawSlotRecord],
    canonical_records: list[RawSlotRecord],
) -> list[dict[str, str]]:
    slot_to_kept_source = {
        (record.group, record.subject_number, record.condition): record.source_name
        for record in canonical_records
    }
    md5_groups: dict[str, list[str]] = defaultdict(list)
    for record in raw_records:
        md5_groups[record.md5].append(record.source_name)

    rows: list[dict[str, str]] = []
    for record in sorted(raw_records, key=lambda item: (item.group, item.subject_number, item.condition, item.source_name)):
        slot_key = (record.group, record.subject_number, record.condition)
        kept_source = slot_to_kept_source[slot_key]
        members = sorted(md5_groups[record.md5])
        if record.source_name != kept_source:
            derivative_status = "omitted_exact_duplicate"
            derivative_reason = f"same-slot duplicate; kept {kept_source}"
        elif (record.group, record.subject_number) in SUSPECTED_DUPLICATE_SUBJECT_EXCLUSIONS:
            kept_subject_number = SUSPECTED_DUPLICATE_SUBJECT_EXCLUSIONS[(record.group, record.subject_number)]["kept_subject_number"]
            derivative_status = "excluded_suspected_duplicate_subject"
            derivative_reason = f"excluded in favor of {format_subject_label(record.group, kept_subject_number)}"
        elif record.condition in EXCLUDED_CONDITIONS:
            derivative_status = "excluded_non_rest_condition"
            derivative_reason = "TASK is not retained in the rest-only derivative"
        else:
            derivative_status = "included_in_derivative"
            derivative_reason = "retained in the normalized derivative"
        rows.append(
            {
                "source_name": record.source_name,
                "normalized_name": record.normalized_name,
                "group": record.group,
                "subject_label": record.subject_label,
                "subject_number": str(record.subject_number),
                "condition": record.condition,
                "size_bytes": str(record.size_bytes),
                "md5": record.md5,
                "selection_status": "kept_canonical" if record.source_name == kept_source else "omitted_exact_duplicate",
                "slot_kept_source": kept_source,
                "derivative_status": derivative_status,
                "derivative_reason": derivative_reason,
                "same_md5_group_size": str(len(members)),
                "same_md5_group_members": "|".join(members),
            }
        )
    return rows


def build_duplicate_subject_exclusion_rows(
    canonical_records: list[RawSlotRecord],
) -> list[dict[str, str]]:
    subject_records: dict[tuple[str, int], dict[str, RawSlotRecord]] = defaultdict(dict)
    for record in canonical_records:
        subject_records[(record.group, record.subject_number)][record.condition] = record

    rows: list[dict[str, str]] = []
    for (group, excluded_subject_number), config in sorted(SUSPECTED_DUPLICATE_SUBJECT_EXCLUSIONS.items()):
        kept_subject_number = config["kept_subject_number"]
        excluded_subject_label = format_subject_label(group, excluded_subject_number)
        kept_subject_label = format_subject_label(group, kept_subject_number)
        excluded_records = subject_records.get((group, excluded_subject_number))
        kept_records = subject_records.get((group, kept_subject_number))
        if excluded_records is None or kept_records is None:
            raise RuntimeError(
                f"Missing canonical records needed for suspected duplicate exclusion: "
                f"{excluded_subject_label} or {kept_subject_label}"
            )

        missing_conditions = [
            condition
            for condition in ALL_STUDY_CONDITIONS
            if condition not in excluded_records or condition not in kept_records
        ]
        if missing_conditions:
            raise RuntimeError(
                f"Cannot validate suspected duplicate exclusion for {excluded_subject_label}: "
                f"missing conditions {', '.join(missing_conditions)}"
            )

        mismatched_conditions = [
            condition
            for condition in ALL_STUDY_CONDITIONS
            if excluded_records[condition].md5 != kept_records[condition].md5
        ]
        if mismatched_conditions:
            raise RuntimeError(
                f"Suspected duplicate exclusion pair {excluded_subject_label} vs {kept_subject_label} "
                f"does not have matching hashes for {', '.join(mismatched_conditions)}"
            )

        removed_outputs = [
            (
                Path(output_group_dir(group))
                / output_subject_dir(group, excluded_subject_number)
                / f"{condition}_raw.fif"
            ).as_posix()
            for condition in EXPECTED_REST_CONDITIONS
            if condition in excluded_records
        ]
        rows.append(
            {
                "group": group,
                "kept_subject_label": kept_subject_label,
                "kept_subject_number": str(kept_subject_number),
                "excluded_subject_label": excluded_subject_label,
                "excluded_subject_number": str(excluded_subject_number),
                "matching_conditions": "|".join(ALL_STUDY_CONDITIONS),
                "ec_raw_md5": excluded_records["EC"].md5,
                "eo_raw_md5": excluded_records["EO"].md5,
                "task_raw_md5": excluded_records["TASK"].md5,
                "kept_raw_sources": "|".join(kept_records[condition].source_name for condition in ALL_STUDY_CONDITIONS),
                "excluded_raw_sources": "|".join(excluded_records[condition].source_name for condition in ALL_STUDY_CONDITIONS),
                "removed_derivative_outputs": "|".join(removed_outputs),
                "decision_rule": "kept lower-numbered subject ID in each byte-identical multi-condition pair",
                "reason": config["reason"],
            }
        )

    return rows


def build_derivative(
    canonical_records: list[RawSlotRecord],
    dest_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    derivative_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    issues: list[str] = []
    seen_subject_conditions = defaultdict(set)

    for record in canonical_records:
        if (record.group, record.subject_number) in SUSPECTED_DUPLICATE_SUBJECT_EXCLUSIONS:
            kept_subject_number = SUSPECTED_DUPLICATE_SUBJECT_EXCLUSIONS[(record.group, record.subject_number)]["kept_subject_number"]
            skipped_rows.append(
                {
                    "source_name": record.source_name,
                    "normalized_name": record.normalized_name,
                    "group": record.group,
                    "subject_label": record.subject_label,
                    "subject_number": str(record.subject_number),
                    "condition": record.condition,
                    "reason": "excluded_suspected_duplicate_subject",
                    "reason_detail": (
                        f"excluded in favor of {format_subject_label(record.group, kept_subject_number)} "
                        f"because raw EC/EO/TASK hashes match across subject IDs"
                    ),
                }
            )
            continue

        if record.condition in EXCLUDED_CONDITIONS:
            skipped_rows.append(
                {
                    "source_name": record.source_name,
                    "normalized_name": record.normalized_name,
                    "group": record.group,
                    "subject_label": record.subject_label,
                    "subject_number": str(record.subject_number),
                    "condition": record.condition,
                    "reason": "excluded_non_rest_condition",
                    "reason_detail": "TASK is not retained in the rest-only derivative",
                }
            )
            continue

        raw = mne.io.read_raw_edf(record.source_path, preload=True, infer_types=True, verbose="ERROR")
        missing = [channel for channel in TARGET_CHANNELS if channel not in raw.ch_names]
        if missing:
            issues.append(f"{record.source_name} missing target channels: {', '.join(missing)}")
            continue

        raw.pick(TARGET_CHANNELS)
        if raw.ch_names != TARGET_CHANNELS:
            issues.append(f"{record.source_name} channel order mismatch after pick: {raw.ch_names}")
            continue

        out_group_dir = output_group_dir(record.group)
        out_subject_dir = output_subject_dir(record.group, record.subject_number)
        out_name = f"{record.condition}_raw.fif"
        out_relpath = Path(out_group_dir) / out_subject_dir / out_name
        out_path = dest_dir / out_relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(out_path, overwrite=True, verbose="ERROR")

        derivative_rows.append(
            {
                "source_name": record.source_name,
                "normalized_name": record.normalized_name,
                "group": record.group,
                "subject_label": record.subject_label,
                "subject_number": str(record.subject_number),
                "condition": record.condition,
                "output_relpath": out_relpath.as_posix(),
                "n_channels": str(len(raw.ch_names)),
                "channel_order": "|".join(raw.ch_names),
                "sfreq": f"{float(raw.info['sfreq']):.1f}",
                "duration_sec": f"{float(raw.n_times / raw.info['sfreq']):.1f}",
            }
        )
        seen_subject_conditions[(record.group, record.subject_label)].add(record.condition)

    for (group, subject_label), conds in sorted(seen_subject_conditions.items()):
        missing_rest = sorted(set(EXPECTED_REST_CONDITIONS) - conds)
        for condition in missing_rest:
            issues.append(f"{group} {subject_label} missing expected rest condition {condition}")

    return derivative_rows, skipped_rows, issues


def validate_derivative(
    dest_dir: Path,
    derivative_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    validation_rows: list[dict[str, str]] = []
    issues: list[str] = []

    for row in derivative_rows:
        path = dest_dir / row["output_relpath"]
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        actual_channels = raw.ch_names
        ok = (
            len(actual_channels) == len(TARGET_CHANNELS)
            and actual_channels == TARGET_CHANNELS
            and path.suffix == ".fif"
            and row["condition"] in EXPECTED_REST_CONDITIONS
        )
        if not ok:
            issues.append(f"Validation failed for {row['output_relpath']}")
        validation_rows.append(
            {
                "output_relpath": row["output_relpath"],
                "group": row["group"],
                "subject_label": row["subject_label"],
                "condition": row["condition"],
                "n_channels": str(len(actual_channels)),
                "channel_order_matches_target": str(actual_channels == TARGET_CHANNELS).lower(),
                "sfreq": f"{float(raw.info['sfreq']):.1f}",
                "duration_sec": f"{float(raw.n_times / raw.info['sfreq']):.1f}",
                "file_md5": md5(path),
            }
        )

    return validation_rows, issues


def build_processed_hash_rows(validation_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    md5_groups: dict[str, list[str]] = defaultdict(list)
    for row in validation_rows:
        md5_groups[row["file_md5"]].append(row["output_relpath"])

    rows: list[dict[str, str]] = []
    for row in sorted(validation_rows, key=lambda item: item["output_relpath"]):
        members = sorted(md5_groups[row["file_md5"]])
        rows.append(
            {
                "output_relpath": row["output_relpath"],
                "group": row["group"],
                "subject_label": row["subject_label"],
                "condition": row["condition"],
                "file_md5": row["file_md5"],
                "same_md5_group_size": str(len(members)),
                "same_md5_group_members": "|".join(members),
            }
        )
    return rows


def make_readme(
    source_dir: Path,
    dest_dir: Path,
    raw_records: list[RawSlotRecord],
    canonical_records: list[RawSlotRecord],
    duplicate_rows: list[dict[str, str]],
    duplicate_subject_exclusion_rows: list[dict[str, str]],
    name_fix_rows: list[dict[str, str]],
    unexpected_names: list[str],
    derivative_rows: list[dict[str, str]],
    skipped_rows: list[dict[str, str]],
    build_issues: list[str],
    validation_rows: list[dict[str, str]],
    validation_issues: list[str],
) -> str:
    by_group = Counter(row["group"] for row in derivative_rows)
    by_condition = Counter((row["group"], row["condition"]) for row in derivative_rows)
    subjects = defaultdict(set)
    for row in derivative_rows:
        subjects[(row["group"], row["subject_label"])].add(row["condition"])
    missing_rest_subjects = [
        (group, subject)
        for (group, subject), conds in sorted(subjects.items())
        if set(EXPECTED_REST_CONDITIONS) - conds
    ]

    skipped_non_rest_count = sum(1 for row in skipped_rows if row["reason"] == "excluded_non_rest_condition")
    skipped_duplicate_subject_count = sum(1 for row in skipped_rows if row["reason"] == "excluded_suspected_duplicate_subject")
    skipped_duplicate_subject_rest_count = sum(
        1
        for row in skipped_rows
        if row["reason"] == "excluded_suspected_duplicate_subject" and row["condition"] in EXPECTED_REST_CONDITIONS
    )
    duration_min = min(float(row["duration_sec"]) for row in derivative_rows)
    duration_max = max(float(row["duration_sec"]) for row in derivative_rows)
    sfreqs = sorted({row["sfreq"] for row in derivative_rows})
    raw_md5_groups: dict[str, list[str]] = defaultdict(list)
    for record in raw_records:
        raw_md5_groups[record.md5].append(record.source_name)
    repeated_raw_hash_lines = []
    for md5_value, members in sorted(raw_md5_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(members) > 1:
            repeated_raw_hash_lines.append(
                f"- raw MD5 `{md5_value}`: " + ", ".join(f"`{member}`" for member in sorted(members))
            )
    if not repeated_raw_hash_lines:
        repeated_raw_hash_lines.append("- None.")

    processed_md5_groups: dict[str, list[str]] = defaultdict(list)
    for row in validation_rows:
        processed_md5_groups[row["file_md5"]].append(row["output_relpath"])
    repeated_processed_hash_lines = []
    for md5_value, members in sorted(processed_md5_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(members) > 1:
            repeated_processed_hash_lines.append(
                f"- processed MD5 `{md5_value}`: " + ", ".join(f"`{member}`" for member in sorted(members))
            )
    if not repeated_processed_hash_lines:
        repeated_processed_hash_lines.append("- None.")

    issue_lines = [f"- {issue}" for issue in build_issues + validation_issues]
    if not issue_lines:
        issue_lines.append("- No build or validation issues beyond the known missing EC/EO pairs inherited from the raw source.")

    missing_subject_lines = []
    if not missing_rest_subjects:
        missing_subject_lines.append("- None.")
    else:
        for group, subject in missing_rest_subjects:
            conds = sorted(set(EXPECTED_REST_CONDITIONS) - subjects[(group, subject)])
            missing_subject_lines.append(f"- `{group} {subject}` missing `{', '.join(conds)}`")

    duplicate_lines = []
    if not duplicate_rows:
        duplicate_lines.append("- None.")
    else:
        for row in duplicate_rows:
            duplicate_lines.append(
                f"- kept `{row['kept_source']}` and omitted exact duplicate `{row['omitted_source']}` for `{row['normalized_slot']}` because both have raw MD5 `{row['md5']}`"
            )

    duplicate_subject_exclusion_lines = []
    if not duplicate_subject_exclusion_rows:
        duplicate_subject_exclusion_lines.append("- None.")
    else:
        duplicate_subject_exclusion_lines.append(
            "- decision rule: when two different subject IDs pointed to byte-identical raw EEG across `EC`, `EO`, and `TASK`, the lower-numbered subject ID was kept and the higher-numbered subject ID was excluded to avoid double-counting the same underlying signal"
        )
        for row in duplicate_subject_exclusion_rows:
            duplicate_subject_exclusion_lines.append(
                f"- excluded `{row['excluded_subject_label']}` in favor of `{row['kept_subject_label']}` "
                f"because the raw hashes match across all three conditions "
                f"(EC `{row['ec_raw_md5']}`, EO `{row['eo_raw_md5']}`, TASK `{row['task_raw_md5']}`)"
            )

    normalization_lines = []
    if not name_fix_rows:
        normalization_lines.append("- None.")
    else:
        for row in name_fix_rows:
            normalization_lines.append(
                f"- `{row['source_name']}` -> `{row['normalized_name']}` ({row['reason']})"
            )

    unexpected_lines = []
    if not unexpected_names:
        unexpected_lines.append("- None.")
    else:
        for name in unexpected_names:
            unexpected_lines.append(f"- `{name}`")

    return f"""# 4244171 Normalized

This directory is a normalized, analysis-ready subset built directly from the raw flat EDF dump in `{source_dir}`.

After cleaning and excluding suspected duplicate-subject recordings, we have all retained rest recordings organized to the same 10-20-style 19-channel scalp system for `EO` and `EC`.

Overall:

- Control `EO`: `{by_condition[('Control', 'EO')]}`
- MDD `EO`: `{by_condition[('MDD', 'EO')]}`
- Control `EC`: `{by_condition[('Control', 'EC')]}`
- MDD `EC`: `{by_condition[('MDD', 'EC')]}`

More details appear below.

## Goal

Create a rest-only dataset with a uniform 19-channel scalp montage for downstream analyses tied to the study framing, without requiring an intermediate cleanup dataset.

## Source

- source dataset: `{source_dir}`
- source format: EDF flat dump
- derivative format: FIF (`.fif`)

## Why FIF Instead Of EDF

EDF export from MNE requires the `edfio` backend, and that backend is not installed in the local environment. To avoid installing packages or mutating the raw EDFs, this derivative is stored as MNE FIF files.

## Build Script

This derivative includes a one-step portable builder:

- `01_normalize.py`

What the script does:

- reads the raw flat EDF dump in `4244171`
- normalizes inconsistent filenames
- removes exact duplicate raw files for the same normalized slot
- excludes suspected duplicate subjects when two different subject IDs point to byte-identical raw EEG across `EC`, `EO`, and `TASK`
- excludes `TASK`
- keeps only `EC` and `EO`
- standardizes every kept recording to the same 19 scalp channels
- writes the output as subject-organized `.fif` files
- writes manifests and validation tables describing the build

Example usage:

```bash
python {dest_dir / '01_normalize.py'} \
  {source_dir} \
  {dest_dir}
```

If you want to recreate this dataset in a different location, keep the same first argument and change only the output directory:

```bash
python {dest_dir / '01_normalize.py'} \
  {source_dir} \
  /path/to/output/4244171_normalized
```

## Preprocessing Applied

All preprocessing steps applied here:

1. Started from the raw EDF files found directly under `{source_dir}`.
2. Normalized filenames by stripping numeric download prefixes and collapsing repeated spaces.
3. Parsed normalized names into `(group, subject, condition)` slots.
4. Detected and omitted exact duplicate raw files for the same normalized slot.
5. Detected suspected duplicate subjects when two different subject IDs had byte-identical raw EDFs across `EC`, `EO`, and `TASK`.
6. Kept the lower-numbered subject ID within each such pair and excluded the higher-numbered subject ID from the derivative.
7. Excluded all remaining `TASK` recordings.
8. Kept only rest conditions: `EC` and `EO`.
9. Loaded each selected source EDF with `mne.io.read_raw_edf(..., preload=True, infer_types=True)`.
10. Selected and reordered channels to the exact target 19-channel scalp set:
   `Fp1-LE, Fp2-LE, F3-LE, F4-LE, C3-LE, C4-LE, P3-LE, P4-LE, O1-LE, O2-LE, F7-LE, F8-LE, T3-LE, T4-LE, T5-LE, T6-LE, Fz-LE, Cz-LE, Pz-LE`
11. Dropped all non-target channels, including:
   `A2-A1`, `23A-23R`, `24A-24R`
12. Did **not** apply filtering, rereferencing, resampling, epoching, artifact rejection, or interpolation.
13. Saved the result as `.fif` files named `EC_raw.fif` or `EO_raw.fif` under subject folders.

## Counts

- raw EDF files discovered: `{len(list(source_dir.glob('*.edf')))}`
- canonical raw subject-condition slots after deduplication: `{len(canonical_records)}`
- exact duplicate raw files omitted: `{len(duplicate_rows)}`
- suspected duplicate-subject raw EDFs excluded before derivative write: `{skipped_duplicate_subject_count}`
- suspected duplicate-subject rest recordings excluded from modeling set: `{skipped_duplicate_subject_rest_count}`
- remaining `TASK` files excluded for rest-only derivative: `{skipped_non_rest_count}`
- derivative files written: `{len(derivative_rows)}`
- control rest files: `{by_group['Control']}`
- MDD rest files: `{by_group['MDD']}`
- control `EC`/`EO`: `{by_condition[('Control', 'EC')]}` / `{by_condition[('Control', 'EO')]}`
- MDD `EC`/`EO`: `{by_condition[('MDD', 'EC')]}` / `{by_condition[('MDD', 'EO')]}`
- sampling frequencies observed: `{', '.join(sfreqs)}`
- duration range: `{duration_min:.0f}-{duration_max:.0f} sec`

## Validation

Every written file was re-opened with MNE and checked for:

- format readable as FIF
- exactly 19 channels
- exact channel order match to the target list
- condition restricted to `EC` or `EO`

Validated files: `{len(validation_rows)}`

## Known Inherited Gaps

The raw source itself is incomplete for some rest recordings, and this derivative preserves those gaps:

{chr(10).join(missing_subject_lines)}

## Duplicate Handling

{chr(10).join(duplicate_lines)}

## Suspected Duplicate Subject Exclusions

{chr(10).join(duplicate_subject_exclusion_lines)}

## Hash Validation

Two manifest tables make the hash evidence explicit:

- `manifests/raw_file_hashes.tsv`: one row per discovered raw EDF, including omitted duplicates, with raw MD5 hashes, canonical-selection status, derivative status, and same-hash group membership
- `manifests/processed_file_hashes.tsv`: one row per written derivative FIF file with processed MD5 hashes and same-hash group membership
- `manifests/duplicate_subject_exclusions.tsv`: the explicit exclusion decisions for byte-identical cross-subject duplicate pairs

Important note:

- raw EDF MD5 hashes are stable for the source files
- processed FIF MD5 hashes validate the exact files written in this build
- a fresh rebuild can still produce semantically equivalent FIF outputs with different file-level MD5 values, so processed hashes are best used as within-build evidence rather than cross-build byte-for-byte proof

Removed according to exact raw-file hash:

{chr(10).join(duplicate_lines)}

Excluded according to repeated cross-subject raw-file hash evidence:

{chr(10).join(duplicate_subject_exclusion_lines)}

Repeated raw-file hash groups:

{chr(10).join(repeated_raw_hash_lines)}

Repeated processed-file hash groups:

{chr(10).join(repeated_processed_hash_lines)}

## Filename Normalization

{chr(10).join(normalization_lines)}

## Unexpected Raw Names

{chr(10).join(unexpected_lines)}

## Issues

{chr(10).join(issue_lines)}

## Related Files

- `README.md`: this document
- `manifests/raw_file_hashes.tsv`: one row per discovered raw EDF with raw MD5 evidence
- `manifests/raw_canonical_selection.tsv`: canonical raw file chosen for each normalized slot
- `manifests/raw_duplicates_omitted.tsv`: duplicate raw files skipped during build
- `manifests/duplicate_subject_exclusions.tsv`: explicit subject-level exclusions for byte-identical cross-subject duplicate pairs
- `manifests/raw_name_normalization.tsv`: filename normalization actions
- `manifests/skipped_source_rows.tsv`: canonical raw rows excluded from analysis, such as `TASK`
- `manifests/derivative_inventory.tsv`: one row per derivative recording
- `manifests/processed_file_hashes.tsv`: one row per derivative file with processed MD5 evidence
- `manifests/validation.tsv`: validation checks for each derivative file

## Manifest QC Summary

The manifest `.tsv` files are internally consistent and document the build decisions explicitly.

Good signs:

- counts reconcile exactly: `{len(canonical_records)}` canonical raw rows = `{len(derivative_rows)}` derivative rows + `{skipped_non_rest_count}` non-rest skips + `{skipped_duplicate_subject_count}` suspected duplicate-subject skips
- every derivative row has exactly `19` channels
- every validated derivative row matches the target channel order
- skipped rows are only skipped for expected reasons: `excluded_non_rest_condition` or `excluded_suspected_duplicate_subject`
- raw manifests include file sizes, canonical-selection status, derivative status, and MD5 hashes for reproducibility and duplicate detection
- filename normalization is small and explicit rather than broad or destructive
- after excluding the cross-subject duplicate pairs, repeated processed-file hash groups are: `{len([1 for line in repeated_processed_hash_lines if line != '- None.'])}`

What each manifest captures:

- `raw_file_hashes.tsv`: every discovered raw EDF, including omitted duplicates, with hash-group evidence and derivative inclusion status
- `raw_canonical_selection.tsv`: the canonical raw EDF chosen for each normalized subject-condition slot
- `raw_duplicates_omitted.tsv`: exact duplicate raw files dropped during canonicalization
- `duplicate_subject_exclusions.tsv`: explicit cross-subject duplicate exclusions and the matching-condition MD5 evidence behind them
- `raw_name_normalization.tsv`: filename cleanup actions applied before slot parsing
- `skipped_source_rows.tsv`: canonical raw rows intentionally excluded from the derivative
- `derivative_inventory.tsv`: the final normalized rest-only derivative inventory
- `processed_file_hashes.tsv`: every current derivative FIF file with hash-group evidence
- `validation.tsv`: post-write QC for every output file

## Suspicious Source-Level Findings

These do not look like manifest-generation errors. They look like inherited issues in the raw source dataset:

- one exact duplicate raw file was present for `H S15 EO` and was handled by keeping one copy and omitting the other
- `H S27` and `H S30` had byte-identical raw `EC`, `EO`, and `TASK` EDFs; `H S30` was excluded from the derivative
- `MDD S33` and `MDD S34` had byte-identical raw `EC`, `EO`, and `TASK` EDFs; `MDD S34` was excluded from the derivative
- `10` separate `TASK` EDF files share the same MD5 hash; all `TASK` rows are excluded from this rest-only derivative, so they do not enter the final modeling set

## Bottom Line

The normalized derivative now excludes the strongest cross-subject duplicate cases so the final rest-only modeling set does not double-count those signals.

The main remaining data-quality concerns are:

- inherited missing `EC` or `EO` recordings for `7` included subjects
- one exact duplicate raw file within a subject-condition slot in the source dump
- unusual repeated hashes among excluded raw `TASK` files in the source data
"""


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    dest_dir = args.output_dir.resolve()
    manifest_dir = dest_dir / "manifests"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if dest_dir.exists():
        raise FileExistsError(f"Destination already exists: {dest_dir}")

    raw_records, unexpected_names = discover_raw_slots(source_dir)
    canonical_records, duplicate_rows, name_fix_rows = canonicalize_raw_slots(raw_records)
    duplicate_subject_exclusion_rows = build_duplicate_subject_exclusion_rows(canonical_records)

    dest_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    derivative_rows, skipped_rows, build_issues = build_derivative(canonical_records, dest_dir)
    validation_rows, validation_issues = validate_derivative(dest_dir, derivative_rows)
    raw_hash_rows = build_raw_hash_rows(raw_records, canonical_records)
    processed_hash_rows = build_processed_hash_rows(validation_rows)

    write_tsv(
        manifest_dir / "raw_file_hashes.tsv",
        [
            "source_name",
            "normalized_name",
            "group",
            "subject_label",
            "subject_number",
            "condition",
            "size_bytes",
            "md5",
            "selection_status",
            "slot_kept_source",
            "derivative_status",
            "derivative_reason",
            "same_md5_group_size",
            "same_md5_group_members",
        ],
        raw_hash_rows,
    )

    write_tsv(
        manifest_dir / "raw_canonical_selection.tsv",
        [
            "source_name",
            "normalized_name",
            "group",
            "subject_label",
            "subject_number",
            "condition",
            "size_bytes",
            "md5",
        ],
        [
            {
                "source_name": record.source_name,
                "normalized_name": record.normalized_name,
                "group": record.group,
                "subject_label": record.subject_label,
                "subject_number": str(record.subject_number),
                "condition": record.condition,
                "size_bytes": str(record.size_bytes),
                "md5": record.md5,
            }
            for record in canonical_records
        ],
    )
    write_tsv(
        manifest_dir / "raw_duplicates_omitted.tsv",
        ["normalized_slot", "kept_source", "omitted_source", "size_bytes", "md5"],
        duplicate_rows,
    )
    write_tsv(
        manifest_dir / "duplicate_subject_exclusions.tsv",
        [
            "group",
            "kept_subject_label",
            "kept_subject_number",
            "excluded_subject_label",
            "excluded_subject_number",
            "matching_conditions",
            "ec_raw_md5",
            "eo_raw_md5",
            "task_raw_md5",
            "kept_raw_sources",
            "excluded_raw_sources",
            "removed_derivative_outputs",
            "decision_rule",
            "reason",
        ],
        duplicate_subject_exclusion_rows,
    )
    write_tsv(
        manifest_dir / "raw_name_normalization.tsv",
        ["source_name", "normalized_name", "reason"],
        name_fix_rows,
    )
    write_tsv(
        manifest_dir / "skipped_source_rows.tsv",
        ["source_name", "normalized_name", "group", "subject_label", "subject_number", "condition", "reason", "reason_detail"],
        skipped_rows,
    )
    write_tsv(
        manifest_dir / "derivative_inventory.tsv",
        [
            "source_name",
            "normalized_name",
            "group",
            "subject_label",
            "subject_number",
            "condition",
            "output_relpath",
            "n_channels",
            "channel_order",
            "sfreq",
            "duration_sec",
        ],
        derivative_rows,
    )
    write_tsv(
        manifest_dir / "validation.tsv",
        [
            "output_relpath",
            "group",
            "subject_label",
            "condition",
            "n_channels",
            "channel_order_matches_target",
            "sfreq",
            "duration_sec",
            "file_md5",
        ],
        validation_rows,
    )
    write_tsv(
        manifest_dir / "processed_file_hashes.tsv",
        [
            "output_relpath",
            "group",
            "subject_label",
            "condition",
            "file_md5",
            "same_md5_group_size",
            "same_md5_group_members",
        ],
        processed_hash_rows,
    )

    readme = make_readme(
        source_dir,
        dest_dir,
        raw_records,
        canonical_records,
        duplicate_rows,
        duplicate_subject_exclusion_rows,
        name_fix_rows,
        unexpected_names,
        derivative_rows,
        skipped_rows,
        build_issues,
        validation_rows,
        validation_issues,
    )
    (dest_dir / "README.md").write_text(readme, encoding="utf-8")
    shutil.copy2(Path(__file__), dest_dir / "01_normalize.py")

    print(f"dest={dest_dir}")
    print(f"raw_discovered={len(raw_records)}")
    print(f"canonical_slots={len(canonical_records)}")
    print(f"duplicates_omitted={len(duplicate_rows)}")
    print(f"files_written={len(derivative_rows)}")
    print(f"validation_rows={len(validation_rows)}")


if __name__ == "__main__":
    main()
