# MDD — Mumtaz et al. 2017 (Figshare)

Major depressive disorder vs healthy controls. Arrives as raw `.mat` files
per subject — needs conversion to BIDS before the pipeline can read it.

Two task conditions are extracted:
- MDD-EC (eyes-closed resting)
- MDD-EO (eyes-open resting)

## Download

DOI: `10.6084/m9.figshare.4244171`
Page: https://figshare.com/articles/dataset/EEG_Data_New/4244171

Download the bundle and unpack to e.g. `/your/local/mdd_raw/`.

## Convert to BIDS

```bash
python data_sources/mdd/build_bids.py \
    --in  /your/local/mdd_raw/ \
    --out /your/local/mdd_bids/
```

(Conversion script coming — placeholder for now.)

## Validate

```bash
python data_sources/mdd/validate.py /your/local/mdd_bids/
```

Expected after curation:
- MDD-EC: 56 subjects (29 disease, 27 control), 10978 total epochs
- MDD-EO: 59 subjects (31 disease, 28 control), 11539 total epochs

## After conversion

Update the path placeholder in `../../configs/mdd_ec/{lpso,w_c}/*.yaml`
and `../../configs/mdd_eo/{lpso,w_c}/*.yaml` to point at
`/your/local/mdd_bids/sub-XXX/eeg/...`.
