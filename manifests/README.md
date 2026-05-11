# Split Manifests — held-out subject IDs per fold

This directory contains the canonical fold definitions used by every result
reported in the paper. Ten CSV files, one per (cohort × P) combination:

```
ad_p6_folds.csv      ad_p2_folds.csv
ftd_p6_folds.csv     ftd_p2_folds.csv
mdd_ec_p6_folds.csv  mdd_ec_p2_folds.csv
mdd_eo_p6_folds.csv  mdd_eo_p2_folds.csv
scz_p6_folds.csv     scz_p2_folds.csv
```

## Format

Each file has **50 rows** (one per fold) and **P+1 columns**:

```
fold, sub_1, sub_2, ..., sub_P
```

- `fold` is 1–50 (canonical ordering, see below).
- `sub_i` columns hold the held-out test subject IDs for that fold.
- Within each row, subjects are sorted ascending by ID.

Example — first 3 rows of `ad_p6_folds.csv`:

```csv
fold,sub_1,sub_2,sub_3,sub_4,sub_5,sub_6
1,1,6,11,50,59,64
2,1,13,29,40,45,53
3,1,16,30,49,55,56
```

The paper's Table 10 shows the first six rows of `ad_p6_folds.csv` exactly,
giving the AD P=6 LPSO setup that produced every AD result reported at P=6.

## Subject IDs

Subject IDs are integers matching the BIDS naming in each cohort's source
dataset (`sub-001`, `sub-002`, …). Across cohorts the integer ranges are:

| Cohort | Subject ID range                   | Disease ↔ Control split    |
| ------ | ---------------------------------- | --------------------------- |
| AD     | 1–65                               | 1–36 (alz) / 37–65 (ctrl)   |
| FTD    | 1–52 with FTD-specific re-indexing | from OpenNeuro ds004904     |
| MDD-EC | 1–56                               | mdd / cntrl per Mumtaz 2017 |
| MDD-EO | 1–59                               | mdd / cntrl per Mumtaz 2017 |
| SCZ    | 1–28                               | 1–14 (scz) / 15–28 (ctrl)   |

## Fold ordering

Rows are sorted by `(sub_1, sub_2, …, sub_P)` as an integer tuple in
ascending order. This means fold 1 contains the lowest-indexed subjects.

The same 50 folds are used for both the T-test pipeline and the PCA
pipeline of each (cohort, P) — that's why only one manifest exists per
(cohort, P) rather than per (cohort, P, pipeline).

## How these were generated

The LPSO Random-50 design draws P subjects uniformly at random per fold,
across 50 unique folds at both P=6 and P=2, separately per cohort. The
seed (42) and draw logic live in the pipeline source (`pipeline/` directory
when released). These CSVs are the resulting subject lists, derived from
the canonical per-fold result tables in the project's data outputs.

## How to reproduce a single fold's results

Pick a row, train your model on all OTHER subjects, evaluate on the listed
P subjects. The accuracies you'll get should match what's in the paper's
appendix range tables if you are using the same machine learning model and pre-processing techniques.
