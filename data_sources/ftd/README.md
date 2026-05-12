# FTD — OpenNeuro `ds004504` (Miltiadous et al. 2023)

Frontotemporal dementia vs healthy controls.

⚠️ **Note for camera-ready:** The submitted paper text and Table 1 cite FTD
as coming from `ds004904`. That dataset ID does not exist — the FTD cohort
actually comes from the same `ds004504` as the AD cohort. The FTD subset
is `Group == "F"` in that dataset (23 subjects). The error is consistent
across the paper but does not affect any numerical results, since all
subject IDs in the manifests are valid FTD subjects from `ds004504`.

The dataset is checked out as a git submodule at
[`../ds004504/`](../ds004504/), via
[OpenNeuroDatasets/ds004504](https://github.com/OpenNeuroDatasets/ds004504).
Cloning the submodule fetches only the small git-annex pointer files — see
the parent README for instructions on how to fetch the actual EEG `.set`
files.

## Subject selection

After cloning, FTD subjects are everything with `Group == "F"` in
`../ds004504/participants.tsv`. Controls (shared with AD) are
`Group == "C"`.

## Validate

```bash
python data_sources/validate.py --cohort ftd --bids ../ds004504
```

Expected: 23 FTD + 29 controls = 52 subjects; 25881 total epochs after
preprocessing.

## After download

Update the path placeholder in `../../configs/ftd/{lpso,w_c}/*.yaml` to
point at `/your/local/checkout/data_sources/ds004504/sub-XXX/eeg/...`.
