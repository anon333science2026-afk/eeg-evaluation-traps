# `data_sources/` — how to obtain and prepare the raw EEG data

This repository ships configs, manifests, and processed canonical data — it
does **not** ship the raw EEG recordings themselves. Each cohort has its own
subdirectory here describing how to obtain and (if needed) convert the raw
data into BIDS format.

| Cohort | Source                               | Format on arrival | Needs conversion?  |
| ------ | ------------------------------------ | ----------------- | ------------------ |
| AD     | OpenNeuro `ds004504` (Group="A")     | BIDS              | No — download only |
| FTD    | OpenNeuro `ds004504` (Group="F")\*   | BIDS              | No — download only |
| MDD    | Figshare (Mumtaz et al. 2017)        | raw `.mat`        | Yes — see `mdd/`   |
| SCZ    | RepOD (Olejarczyk & Jernajczyk 2017) | raw `.edf`        | Yes — see `scz/`   |

\*The submitted paper cites `ds004904` for FTD; that ID does not exist and is
a typo to be corrected for camera-ready. AD and FTD ship together in
`ds004504` (Miltiadous et al. 2023). See `ftd/README.md`.

After running each cohort's steps, the data should live at a path of your
choosing (e.g. `/your/local/<cohort>/...`). Update the path placeholder in
the YAML configs under `../configs/<cohort>/` to point at your local copy
before running the pipeline.

## OpenNeuro data via git submodule

The AD/FTD dataset (`ds004504`) is tracked as a git submodule at
[`ds004504/`](ds004504/). To fetch the dataset's metadata (small git-annex
pointer files):

```bash
# If you didn't clone with --recursive:
git submodule update --init data_sources/ds004504
```

To then fetch the actual EEG `.set` files (the dataset is DataLad-managed
on OpenNeuro), use one of:

```bash
# Option 1: DataLad get
cd data_sources/ds004504
datalad get .

# Option 2: aws s3 sync (overwrites the submodule with the full dataset)
# *NOTICE* option 2 worked the best for me especially on Mac OS and HPC. The others were diffucult to work reliably.
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 data_sources/ds004504

# Option 3: openneuro-py CLI
openneuro-py download --dataset=ds004504 --target-dir=data_sources/ds004504
```

Project page: https://openneuro.org/datasets/ds004504

## Expected per-cohort counts

Subjects / disease / control :

- AD: 65 / 36 / 29
- FTD: 52 / 23 / 29
- MDD-EC: 56 / 29 / 27
- MDD-EO: 59 / 31 / 28
- SCZ: 28 / 14 / 14
