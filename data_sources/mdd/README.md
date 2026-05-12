# MDD — Mumtaz et al. 2017 (Figshare)

Source: DOI [`10.6084/m9.figshare.4244171`](https://doi.org/10.6084/m9.figshare.4244171) —
ships as a flat dump of `.edf` files (one per subject × condition,
e.g. `H S1 EC.edf`, `MDD S1 EC.edf`, `… EO.edf`, `… TASK.edf`).

Two task conditions are extracted (TASK is dropped, only resting-state
used): MDD-EC (eyes-closed) and MDD-EO (eyes-open).

## Download

Figshare hosts the dataset under a "Download all" button — no CLI tooling
or stable AWS/S3 mirror is available, so the download is manual via web
browser:

  https://figshare.com/articles/dataset/EEG_Data_New/4244171

Unpack the archive to a flat directory of `.edf` files. After unpacking
you should have ~177 EDFs (3 conditions — EC / EO / TASK — across ~59
subjects). Pass that directory as the `RAW=` argument to stage 1 below.

## Scripts

```text
data_sources/mdd/
├── README.md
├── 01_normalize.py     raw .edf flat dump → normalized FIF derivative
├── 02_to_bids.py       normalized FIF → BIDS-formatted EDF tree
└── validate.py         confirm BIDS tree matches paper Table 1
```

MDD curation is **sequential** (`01_normalize.py` then `02_to_bids.py`),
unlike SCZ where the two stages are parallel.

**Stage 1 (`01_normalize.py`):** read each raw EDF, enforce the 10-20
canonical channel set, drop the TASK condition, harmonize channel
ordering across subjects, write `H_S<N>/EC_raw.fif` and `H_S<N>/EO_raw.fif`
(controls) / `MDD_S<N>/...` (cases). Also handles two byte-identical
duplicate-subject exclusions documented in the source dataset
(Control 30 ≡ Control 27; MDD 34 ≡ MDD 33).

**Stage 2 (`02_to_bids.py`):** wrap the normalized FIF derivative into
BIDS layout (`sub-001 … sub-061/eeg/sub-XXX_task-{eyesclosed,eyesopen}_eeg.edf`
plus sidecars + `participants.tsv`).

## Run

```bash
RAW=/your/local/path/to/4244171               # the flat raw EDF dump
NORM=/your/local/path/to/mdd_normalized       # stage 1 output
BIDS=/your/local/path/to/mdd_bids             # stage 2 output

# Stage 1: raw → normalized FIF derivative
python data_sources/mdd/01_normalize.py "$RAW" "$NORM"

# Stage 2: normalized → BIDS-formatted EDF tree
python data_sources/mdd/02_to_bids.py --source-root "$NORM" --output-root "$BIDS" --clean --overwrite

# Validate the BIDS output:
python data_sources/mdd/validate.py "$BIDS"
```

Expected after curation:
- 61 total subjects in `participants.tsv` (32 mdd, 29 control)
- MDD-EC: 56 subjects with `task-eyesclosed_eeg.edf` (29 mdd, 27 control)
- MDD-EO: 59 subjects with `task-eyesopen_eeg.edf`   (31 mdd, 28 control)
- Pipeline-side preprocessing produces 10978 EC + 11539 EO epochs (paper Table 1)

After conversion, update `../../configs/mdd_ec/{lpso,w_c}/*.yaml` and
`../../configs/mdd_eo/{lpso,w_c}/*.yaml` path placeholders to point at
`/your/local/mdd_bids/sub-XXX/eeg/sub-XXX_task-{eyesclosed,eyesopen}_eeg.edf`.
