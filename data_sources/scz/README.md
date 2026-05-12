# SCZ — Olejarczyk & Jernajczyk 2017 (RepOD)

Schizophrenia vs healthy controls. Arrives as raw `.edf` files per subject —
needs minimal restructuring to fit BIDS before the pipeline can read it.

## Download

DOI: `10.18150/repod.0107441`
Page: http://dx.doi.org/10.18150/repod.0107441

Download the archive and unpack to e.g. `/your/local/scz_raw/`.

## Convert to BIDS

```bash
python data_sources/scz/build_bids.py \
    --in  /your/local/scz_raw/ \
    --out /your/local/scz_bids/
```

(Conversion script coming — placeholder for now.)

## Validate

```bash
python data_sources/scz/validate.py /your/local/scz_bids/
```

Expected after curation:
- 28 subjects (14 disease, 14 control)
- 18535 total epochs (after preprocessing — the validator checks subject
  counts only; epoch totals are reached after the preprocessing step in
  `../../configs/scz/`)

## After conversion

Update the path placeholder in `../../configs/scz/{lpso,w_c}/*.yaml` to
point at `/your/local/scz_bids/sub-XXX/eeg/sub-XXX_task-rest_eeg.edf`.
