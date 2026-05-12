#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Iterable

import mne


BIDS_VERSION = "1.11.1"
GENERATED_SCRIPT_NAME = "02_to_bids.py"
GROUP_DIRS = (
    ("healthy_controls", "control"),
    ("major_depressive_disorder", "mdd"),
)
CONDITION_TO_TASK = {
    "EC": "eyesclosed",
    "EO": "eyesopen",
}
GENERATED_ROOT_FILES = (
    ".bidsignore",
    "CHANGES",
    "README",
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
)


@dataclass(frozen=True)
class InventoryRecord:
    source_name: str
    normalized_name: str
    group: str
    subject_label: str
    subject_number: int
    condition: str
    output_relpath: Path
    n_channels: int
    channel_order: tuple[str, ...]
    sfreq: float
    duration_sec: float


@dataclass(frozen=True)
class RecordingRecord:
    condition: str
    task_name: str
    source_path: Path
    source_relpath: Path
    source_name: str
    normalized_name: str
    n_channels: int
    channel_order: tuple[str, ...]
    sfreq: float
    duration_sec: float


@dataclass
class SubjectRecord:
    participant_id: str
    group: str
    original_subject_id: str
    source_group_folder: str
    source_subject_dir: Path
    subject_number: int
    recordings: dict[str, RecordingRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert 4244171_normalized into a BIDS-organized resting-state EEG dataset "
            "with EDF recordings."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root of the normalized source dataset. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="BIDS output root. Defaults to <source-root>/BIDS.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously generated BIDS subject folders and generated root files before writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files if they already exist.",
    )
    return parser.parse_args()


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def parse_output_subject_dir(subject_dir_name: str) -> tuple[str, int]:
    control_match = re.fullmatch(r"H_S(\d+)", subject_dir_name)
    if control_match:
        return ("control", int(control_match.group(1)))
    mdd_match = re.fullmatch(r"MDD_S(\d+)", subject_dir_name)
    if mdd_match:
        return ("mdd", int(mdd_match.group(1)))
    raise ValueError(f"Unexpected normalized subject directory: {subject_dir_name}")


def parse_source_subject_dir(subject_dir_name: str) -> int:
    match = re.search(r"(\d+)$", subject_dir_name)
    if not match:
        raise ValueError(f"Could not parse numeric subject suffix from {subject_dir_name}")
    return int(match.group(1))


def participant_id(index: int) -> str:
    return f"sub-{index:03d}"


def bool_to_yes_no(value: bool) -> str:
    return "yes" if value else "no"


def output_relpath(participant_label: str, task_name: str) -> Path:
    return Path(participant_label) / "eeg" / f"{participant_label}_task-{task_name}_eeg.edf"


def root_generated_path(output_root: Path, relpath: Path) -> Path:
    return output_root / relpath


def load_inventory(source_root: Path) -> dict[tuple[str, int, str], InventoryRecord]:
    manifest_path = source_root / "manifests" / "derivative_inventory.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing derivative inventory manifest: {manifest_path}")

    inventory: dict[tuple[str, int, str], InventoryRecord] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_relpath = Path(row["output_relpath"])
            subject_dir_name = source_relpath.parts[1]
            group, subject_number = parse_output_subject_dir(subject_dir_name)
            condition = row["condition"]
            key = (group, subject_number, condition)
            inventory[key] = InventoryRecord(
                source_name=row["source_name"],
                normalized_name=row["normalized_name"],
                group=row["group"],
                subject_label=row["subject_label"],
                subject_number=int(row["subject_number"]),
                condition=condition,
                output_relpath=source_relpath,
                n_channels=int(float(row["n_channels"])),
                channel_order=tuple(row["channel_order"].split("|")),
                sfreq=float(row["sfreq"]),
                duration_sec=float(row["duration_sec"]),
            )
    return inventory


def discover_subjects(source_root: Path, inventory: dict[tuple[str, int, str], InventoryRecord]) -> list[SubjectRecord]:
    discovered: list[tuple[str, str, Path, int]] = []
    for source_group_folder, group in GROUP_DIRS:
        group_root = source_root / source_group_folder
        if not group_root.exists():
            raise FileNotFoundError(f"Missing group directory: {group_root}")
        subject_dirs = sorted(
            [path for path in group_root.iterdir() if path.is_dir()],
            key=lambda path: parse_source_subject_dir(path.name),
        )
        for subject_dir in subject_dirs:
            discovered.append(
                (
                    source_group_folder,
                    group,
                    subject_dir,
                    parse_source_subject_dir(subject_dir.name),
                )
            )

    subject_records: list[SubjectRecord] = []
    for index, (source_group_folder, group, subject_dir, subject_number) in enumerate(discovered, start=1):
        canonical_participant_id = participant_id(index)
        recordings: dict[str, RecordingRecord] = {}
        for condition, task_name in CONDITION_TO_TASK.items():
            source_path = subject_dir / f"{condition}_raw.fif"
            if not source_path.exists():
                continue
            key = (group, subject_number, condition)
            inventory_record = inventory.get(key)
            if inventory_record is None:
                raise KeyError(
                    f"Could not find manifest metadata for subject {subject_dir.name} condition {condition}"
                )
            recordings[condition] = RecordingRecord(
                condition=condition,
                task_name=task_name,
                source_path=source_path,
                source_relpath=inventory_record.output_relpath,
                source_name=inventory_record.source_name,
                normalized_name=inventory_record.normalized_name,
                n_channels=inventory_record.n_channels,
                channel_order=inventory_record.channel_order,
                sfreq=inventory_record.sfreq,
                duration_sec=inventory_record.duration_sec,
            )

        subject_records.append(
            SubjectRecord(
                participant_id=canonical_participant_id,
                group=group,
                original_subject_id=subject_dir.name,
                source_group_folder=source_group_folder,
                source_subject_dir=subject_dir,
                subject_number=subject_number,
                recordings=recordings,
            )
        )
    return subject_records


def cleanup_generated_output(output_root: Path) -> None:
    if not output_root.exists():
        return

    for child in output_root.iterdir():
        if child.is_dir() and re.fullmatch(r"sub-\d{3}", child.name):
            shutil.rmtree(child)

    for file_name in GENERATED_ROOT_FILES:
        path = output_root / file_name
        if path.exists():
            path.unlink()

    sourcedata_csv = output_root / "sourcedata" / "subject_metadata.csv"
    if sourcedata_csv.exists():
        sourcedata_csv.unlink()


def ensure_output_root(output_root: Path, clean: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if clean:
        cleanup_generated_output(output_root)
    (output_root / "sourcedata").mkdir(parents=True, exist_ok=True)


def export_recording_to_edf(source_path: Path, destination_path: Path, overwrite: bool) -> None:
    if destination_path.exists() and not overwrite:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_fif(source_path, preload=False, verbose="ERROR")
    mne.export.export_raw(destination_path, raw, fmt="edf", overwrite=True)


def eeg_sidecar_payload(recording: RecordingRecord) -> dict:
    return {
        "TaskName": recording.task_name,
        "SamplingFrequency": recording.sfreq,
        "PowerLineFrequency": "n/a",
        "SoftwareFilters": "n/a",
        "EEGReference": "Linked ears (LE), inferred from source channel names ending in '-LE'.",
        "RecordingType": "continuous",
        "EEGChannelCount": recording.n_channels,
        "EEGPlacementScheme": "International 10-20 system (19-channel subset)",
    }


def channels_rows(recording: RecordingRecord) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for channel_name in recording.channel_order:
        rows.append(
            {
                "name": channel_name,
                "type": "EEG",
                "units": "V",
                "sampling_frequency": f"{recording.sfreq:g}",
                "reference": "LE",
                "status": "good",
                "status_description": "n/a",
            }
        )
    return rows


def participants_rows(subject_records: Iterable[SubjectRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subject in subject_records:
        rows.append(
            {
                "participant_id": subject.participant_id,
                "group": subject.group,
                "original_subject_id": subject.original_subject_id,
                "has_eyesclosed": bool_to_yes_no("EC" in subject.recordings),
                "has_eyesopen": bool_to_yes_no("EO" in subject.recordings),
            }
        )
    return rows


def subject_metadata_rows(subject_records: Iterable[SubjectRecord], output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subject in subject_records:
        ec = subject.recordings.get("EC")
        eo = subject.recordings.get("EO")
        rows.append(
            {
                "participant_id": subject.participant_id,
                "group": subject.group,
                "original_subject_id": subject.original_subject_id,
                "source_group_folder": subject.source_group_folder,
                "source_subject_dir": str(subject.source_subject_dir),
                "has_eyesclosed": bool_to_yes_no(ec is not None),
                "has_eyesopen": bool_to_yes_no(eo is not None),
                "eyesclosed_source_raw_name": ec.source_name if ec else "",
                "eyesclosed_source_normalized_name": ec.normalized_name if ec else "",
                "eyesclosed_source_derivative_relpath": str(ec.source_relpath) if ec else "",
                "eyesclosed_source_derivative_abspath": str(ec.source_path) if ec else "",
                "eyesclosed_bids_relpath": str(output_relpath(subject.participant_id, ec.task_name)) if ec else "",
                "eyesclosed_sfreq_hz": f"{ec.sfreq:g}" if ec else "",
                "eyesclosed_duration_sec": f"{ec.duration_sec:g}" if ec else "",
                "eyesopen_source_raw_name": eo.source_name if eo else "",
                "eyesopen_source_normalized_name": eo.normalized_name if eo else "",
                "eyesopen_source_derivative_relpath": str(eo.source_relpath) if eo else "",
                "eyesopen_source_derivative_abspath": str(eo.source_path) if eo else "",
                "eyesopen_bids_relpath": str(output_relpath(subject.participant_id, eo.task_name)) if eo else "",
                "eyesopen_sfreq_hz": f"{eo.sfreq:g}" if eo else "",
                "eyesopen_duration_sec": f"{eo.duration_sec:g}" if eo else "",
            }
        )
    return rows


def participants_json_payload() -> dict:
    return {
        "participant_id": {
            "Description": "Canonical BIDS participant identifier assigned during conversion.",
        },
        "group": {
            "Description": "Diagnostic group preserved from the normalized source derivative.",
            "Levels": {
                "control": "Healthy control participant.",
                "mdd": "Major depressive disorder participant.",
            },
        },
        "original_subject_id": {
            "Description": "Original subject identifier from the normalized source derivative.",
        },
        "has_eyesclosed": {
            "Description": "Whether an eyes-closed resting-state recording is present for this participant.",
            "Levels": {
                "yes": "Eyes-closed recording present.",
                "no": "Eyes-closed recording absent.",
            },
        },
        "has_eyesopen": {
            "Description": "Whether an eyes-open resting-state recording is present for this participant.",
            "Levels": {
                "yes": "Eyes-open recording present.",
                "no": "Eyes-open recording absent.",
            },
        },
    }


def dataset_description_payload(source_root: Path) -> dict:
    return {
        "Name": "4244171 Normalized Rest EEG BIDS Conversion",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "GeneratedBy": [
            {
                "Name": GENERATED_SCRIPT_NAME,
                "Version": "1.0.0",
                "Description": (
                    "Converts the normalized 4244171 rest-only FIF derivative into an "
                    "EEG-BIDS-style EDF dataset."
                ),
            }
        ],
        "SourceDatasets": [
            {
                "URL": source_root.as_uri(),
                "Description": (
                    "Normalized rest-only derivative with EO/EC FIF files built from the "
                    "flat 4244171 EDF dump."
                ),
            }
        ],
        "License": "n/a",
    }


def build_readme(subject_records: list[SubjectRecord]) -> str:
    control_count = sum(1 for subject in subject_records if subject.group == "control")
    mdd_count = sum(1 for subject in subject_records if subject.group == "mdd")
    ec_count = sum(1 for subject in subject_records if "EC" in subject.recordings)
    eo_count = sum(1 for subject in subject_records if "EO" in subject.recordings)
    incomplete = [
        subject
        for subject in subject_records
        if not ("EC" in subject.recordings and "EO" in subject.recordings)
    ]

    lines = [
        "# 4244171 Normalized BIDS Conversion",
        "",
        "This directory is a BIDS-organized conversion of the normalized 4244171 resting-state EEG derivative.",
        "",
        "Important provenance note:",
        "",
        "- this BIDS dataset was created from `4244171_normalized`, not directly from the raw flat EDF dump",
        "- the source derivative already applied duplicate handling, subject exclusions, rest-only selection, and channel standardization",
        "- because raw EEG-BIDS does not accept FIF as a recording format, the source FIF recordings were exported to EDF during this conversion",
        "",
        "## Subject Mapping Rule",
        "",
        "Canonical BIDS subject IDs were assigned deterministically as:",
        "",
        "1. controls first, ascending by numeric subject number",
        "2. MDD second, ascending by numeric subject number",
        "",
        "Group is stored in `participants.tsv` and is not encoded into the BIDS subject identifier.",
        "",
        "## Counts",
        "",
        f"- controls: `{control_count}`",
        f"- MDD: `{mdd_count}`",
        f"- total participants: `{len(subject_records)}`",
        f"- eyes-closed recordings present: `{ec_count}`",
        f"- eyes-open recordings present: `{eo_count}`",
        f"- participants missing one condition: `{len(incomplete)}`",
        "",
        "## Condition Encoding",
        "",
        "- `EC_raw.fif` -> `task-eyesclosed`",
        "- `EO_raw.fif` -> `task-eyesopen`",
        "",
        "## Metadata Files",
        "",
        "- `participants.tsv`: BIDS-style participant metadata",
        "- `participants.json`: field descriptions for `participants.tsv`",
        "- `sourcedata/subject_metadata.csv`: richer provenance table including original source names and source paths",
        "",
        "## Known Missing Conditions",
        "",
    ]
    if incomplete:
        for subject in incomplete:
            lines.append(
                f"- `{subject.participant_id}` / `{subject.original_subject_id}` (`{subject.group}`): "
                f"eyesclosed={bool_to_yes_no('EC' in subject.recordings)}, "
                f"eyesopen={bool_to_yes_no('EO' in subject.recordings)}"
            )
    else:
        lines.append("- None.")
    return "\n".join(lines)


def build_changes(subject_records: list[SubjectRecord]) -> str:
    today = date.today().isoformat()
    return "\n".join(
        [
            f"{today}",
            "- Created initial BIDS conversion from `4244171_normalized`.",
            "- Exported source FIF recordings into EDF to satisfy accepted raw EEG-BIDS recording formats.",
            "- Assigned canonical participant identifiers with controls first and MDD second.",
            "- Wrote `participants.tsv`, `participants.json`, and `sourcedata/subject_metadata.csv`.",
            f"- Converted {len(subject_records)} participants into the BIDS tree.",
        ]
    )


def build_bidsignore() -> str:
    return "\n".join(
        [
            "plan.md",
            ".DS_Store",
        ]
    )


def export_subject(subject: SubjectRecord, output_root: Path, overwrite: bool) -> None:
    for recording in subject.recordings.values():
        edf_relpath = output_relpath(subject.participant_id, recording.task_name)
        edf_path = root_generated_path(output_root, edf_relpath)
        export_recording_to_edf(recording.source_path, edf_path, overwrite=overwrite)

        base_path = edf_path.with_suffix("")
        eeg_json_path = base_path.with_suffix(".json")
        channels_tsv_path = edf_path.parent / f"{edf_path.stem.replace('_eeg', '')}_channels.tsv"

        write_json(eeg_json_path, eeg_sidecar_payload(recording))
        write_tsv(
            channels_tsv_path,
            ["name", "type", "units", "sampling_frequency", "reference", "status", "status_description"],
            channels_rows(recording),
        )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = (args.output_root or (source_root / "BIDS")).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    mne.set_log_level("ERROR")

    inventory = load_inventory(source_root)
    subject_records = discover_subjects(source_root, inventory)

    ensure_output_root(output_root, clean=args.clean)

    print(f"📦 Source root: {source_root}")
    print(f"📁 Output root: {output_root}")
    print(f"👥 Participants discovered: {len(subject_records)}")
    print(
        f"   Controls: {sum(1 for subject in subject_records if subject.group == 'control')} | "
        f"MDD: {sum(1 for subject in subject_records if subject.group == 'mdd')}"
    )

    for subject in subject_records:
        export_subject(subject, output_root, overwrite=args.overwrite)

    participant_rows = participants_rows(subject_records)
    metadata_rows = subject_metadata_rows(subject_records, output_root)

    write_tsv(
        output_root / "participants.tsv",
        ["participant_id", "group", "original_subject_id", "has_eyesclosed", "has_eyesopen"],
        participant_rows,
    )
    write_json(output_root / "participants.json", participants_json_payload())
    write_json(output_root / "dataset_description.json", dataset_description_payload(source_root))
    write_text(output_root / "README", build_readme(subject_records))
    write_text(output_root / "CHANGES", build_changes(subject_records))
    write_text(output_root / ".bidsignore", build_bidsignore())
    write_csv(
        output_root / "sourcedata" / "subject_metadata.csv",
        [
            "participant_id",
            "group",
            "original_subject_id",
            "source_group_folder",
            "source_subject_dir",
            "has_eyesclosed",
            "has_eyesopen",
            "eyesclosed_source_raw_name",
            "eyesclosed_source_normalized_name",
            "eyesclosed_source_derivative_relpath",
            "eyesclosed_source_derivative_abspath",
            "eyesclosed_bids_relpath",
            "eyesclosed_sfreq_hz",
            "eyesclosed_duration_sec",
            "eyesopen_source_raw_name",
            "eyesopen_source_normalized_name",
            "eyesopen_source_derivative_relpath",
            "eyesopen_source_derivative_abspath",
            "eyesopen_bids_relpath",
            "eyesopen_sfreq_hz",
            "eyesopen_duration_sec",
        ],
        metadata_rows,
    )

    incomplete = [
        subject for subject in subject_records if not ("EC" in subject.recordings and "EO" in subject.recordings)
    ]
    print("✅ BIDS conversion complete.")
    print(f"   participants.tsv rows: {len(participant_rows)}")
    print(f"   metadata csv rows: {len(metadata_rows)}")
    print(f"   participants missing one condition: {len(incomplete)}")


if __name__ == "__main__":
    main()
