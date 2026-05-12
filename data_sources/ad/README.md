# AD — OpenNeuro `ds004504` (Miltiadous et al. 2023)

Alzheimer's disease vs healthy controls. AD and FTD both come from this
single dataset — the AD cohort is the `Group=="A"` subset (36 subjects);
controls are `Group=="C"` (29 subjects).

The dataset is checked out as a git submodule at
[`../ds004504/`](../ds004504/), via
[OpenNeuroDatasets/ds004504](https://github.com/OpenNeuroDatasets/ds004504).
Cloning the submodule fetches only the small git-annex pointer files — see
the parent README for instructions on how to fetch the actual EEG `.set`
files.

## Subject selection

After cloning, AD subjects are everything with `Group == "A"` in
`../ds004504/participants.tsv`. Controls (shared with FTD) are
`Group == "C"`.

## Validate

```bash
python data_sources/validate.py --cohort ad --bids ../ds004504
```

Expected: 36 AD + 29 controls = 65 subjects; 34044 total epochs after
preprocessing.

## After download

Update the path placeholder in `../../configs/ad/{lpso,w_c}/*.yaml`:

```diff
- /path/to/data/ad-ftd/sub-001/eeg/sub-001_task-eyesclosed_eeg.set
+ /your/local/checkout/data_sources/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set
```
