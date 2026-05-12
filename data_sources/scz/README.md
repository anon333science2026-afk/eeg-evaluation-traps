# SCZ — Olejarczyk & Jernajczyk 2017 (RepOD)

Source: DOI [`10.18150/repod.0107441`](http://dx.doi.org/10.18150/repod.0107441) —
raw `.edf` per subject.

## Download

RepOD has no CLI tooling or stable S3 mirror — the download is manual
via web browser. Use the "Access Dataset → Download ZIP" button at:

  https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441

Unpack to a flat directory containing 28 EDF files
(`s01.edf … s14.edf` schizophrenia, `h01.edf … h14.edf` control) plus
RepOD's `MANIFEST.TXT`. Pass that directory as `RAW=` to the scripts
below.

## Scripts

```text
data_sources/scz/
├── README.md
├── scz_utils.py        shared helpers (channel policy, hashing, manifests)
├── 01_normalize.py     build a FIF normalized derivative (analysis-ready)
├── 02_to_bids.py       build a BIDS-formatted derivative (pipeline-ready)
└── validate.py         confirm a BIDS tree matches paper Table 1
```

`01_normalize.py` and `02_to_bids.py` are **independent derivatives** of
the same raw data — not sequential stages. Each starts from the raw EDFs
and applies its own channel-order normalization on the way out. Run
whichever you need.

**Why normalization is needed:** RepOD subjects share the same 19-channel
montage, but the channel order varies across files (one subject —
`h12.edf` — has a different source order than the rest). Without
normalization, downstream feature extraction sees ragged channel ordering.
Both scripts enforce a single canonical channel order on output.

## Run

```bash
RAW=/your/local/path/to/eeg-in-schizophrenia      # the 28 raw .edf files

# Pipeline-ready BIDS tree (what the paper's configs read):
python data_sources/scz/02_to_bids.py "$RAW" /your/local/scz_bids --clean --overwrite

# Optional: analysis-ready FIF derivative (used by some downstream tooling):
python data_sources/scz/01_normalize.py "$RAW" /your/local/scz_normalized --clean --overwrite

# Validate the BIDS output:
python data_sources/scz/validate.py /your/local/scz_bids
```

Expected after curation:
- 28 subjects (14 schizophrenia, 14 control)
- BIDS layout: `sub-001 … sub-028/eeg/sub-XXX_task-rest_eeg.{edf,json}` + `_channels.tsv`
- Pipeline-side preprocessing produces 18535 epochs total (paper Table 1)

After conversion, update `../../configs/scz/{lpso,w_c}/*.yaml` path
placeholders to point at `/your/local/scz_bids/sub-XXX/eeg/sub-XXX_task-rest_eeg.edf`.
