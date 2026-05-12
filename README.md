# EEG Evaluation Traps

Code, configurations, and split manifests for:

> **"Evaluation Traps in EEG Disease Classification: Identity Leakage, Lucky Folds, and Objective Mismatch — Replicated Across AD, FTD, MDD, and SCZ"**

Submitted to NeurIPS 2026.

---

## Status: In Progress

| Component                                                                                  | State          |
| ------------------------------------------------------------------------------------------ | -------------- |
| `configs/` — 120 YAML pipeline configurations (5 cohorts × 24 each)                        | ✅ Released    |
| `manifests/` — held-out subject IDs per fold for every cohort                              | ✅ Released    |
| `processed_data/` — canonical EEG parquets, results CSVs, and W_C summary CSVs                       | ✅ Released    |
| `tables/` — per-table audit framework (recompute every paper table from `processed_data/`)           | ✅ Released    |
| `figures/` — figure regeneration scripts (every paper figure from `processed_data/`)                 | ✅ Released    |
| `tools/` — helpers to rebuild W_C summary CSVs from raw pipeline outputs                   | ✅ Released    |
| `data_sources/` — per-cohort download + curation (ds004504 submodule for AD/FTD; MDD + SCZ BIDS builders) | ✅ Released    |
| `pipeline/` — PySpark + Ray pipeline source                                                | 🚧 Coming soon |

If you need anything in 🚧 status urgently, open an issue.

**Anonymization note.** This repository is double-blind for NeurIPS review. Paths and SLURM account names that referenced specific compute infrastructure have been replaced with placeholders (`/path/to/data/…`, `<your-slurm-account>`). If you spot anything that looks identifying, please open an issue.

---

## Repository layout

```
eeg-evaluation-traps/
├── configs/                    120 YAML pipeline configs
│   ├── ad/{lpso, w_c}/         AD vs control (OpenNeuro ds004504)
│   ├── ftd/{lpso, w_c}/        FTD vs control (OpenNeuro ds004904)
│   ├── mdd_ec/{lpso, w_c}/     MDD vs control, eyes-closed (Mumtaz 2017)
│   ├── mdd_eo/{lpso, w_c}/     MDD vs control, eyes-open (Mumtaz 2017)
│   └── scz/{lpso, w_c}/        SCZ vs control (RepOD Olejarczyk 2017)
├── manifests/                  Held-out subject IDs per fold (10 CSVs)
├── data_sources/               Per-cohort acquisition instructions + ds004504 submodule
├── processed_data/             Canonical EEG data + audit summaries
│   ├── canonical/<cohort>/     Post-rejection epoch parquets (5 cohorts)
│   ├── results/<cohort>/       Combined and fold-level result CSVs
│   ├── per_subject_summaries/  Per-subject epoch counts + top ANOVA features
│   ├── ad_extras/              AD-specific merged fold + prediction CSVs
│   ├── w_c_per_model_accuracies.csv  Pre-computed W_C overlap accuracy
│   └── w_c_subject_acc.csv     Pre-computed W_C majority-vote Sub. Acc.
├── tables/                     Per-table audit framework
│   ├── diff_T<N>.csv           Per-cell (claimed, computed, delta, status)
│   ├── table_T<N>_recomputed.csv  Full recomputed tables
│   └── test-and-build/         check_T<N>_*.py + build_subj_acc_tables.py
├── figures/                    Figure regeneration scripts (19 .py files)
├── tools/                      W_C summary precompute scripts
├── pipeline/                   PySpark + Ray pipeline source (coming soon)
├── bids_builders/              BIDS curation scripts (coming soon)
└── LICENSE                     MIT
```

Per-cohort config layout:

- `lpso/` — 4 configs: `ANOVA_L_6.yaml`, `ANOVA_L_2.yaml`, `PCA_L_6.yaml`, `PCA_L_2.yaml`
- `w_c/` — 20 configs: `ANOVA_W_C_seed{42..51}.yaml` + `PCA_W_C_seed{42..51}.yaml`

---

## Datasets (download separately)

The repository ships **configs and manifests only** — not raw EEG recordings. Download each dataset from its source:

| Cohort | Source                               | Identifier                        |
| ------ | ------------------------------------ | --------------------------------- |
| AD     | OpenNeuro                            | `ds004504`                        |
| FTD    | OpenNeuro                            | `ds004904`                        |
| MDD    | Figshare (Mumtaz et al. 2017)        | DOI `10.6084/m9.figshare.4244171` |
| SCZ    | RepOD (Olejarczyk & Jernajczyk 2017) | DOI `10.18150/repod.0107441`      |

Once downloaded, the data should be organized in BIDS format (the AD/FTD datasets arrive in BIDS; MDD and SCZ require curation — `bids_builders/` will provide the scripts when released).

---

## Using the configs

Each YAML config points at the data via a path placeholder. Two edits are needed before a config will run:

1. **Replace path placeholders.** In any YAML you intend to run, replace `/path/to/data/<cohort>/...` with your actual local data path. For example:

   ```yaml
   - /path/to/data/ad-ftd/sub-001/eeg/sub-001_task-eyesclosed_eeg.set
   ```

   becomes

   ```yaml
   - /your/local/ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set
   ```

2. **Replace the SLURM account placeholder.** Every `slurm_options` block has `--account=<your-slurm-account>`. Replace with your cluster's account or remove if you're not running under SLURM.

The pipeline source itself (once released) reads these YAMLs and runs the full preprocess → feature-extract → LPSO/W_C → ML grid search workflow.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Citation

```
[anonymous authors]. "Evaluation Traps in EEG Disease Classification:
Identity Leakage, Lucky Folds, and Objective Mismatch — Replicated
Across AD, FTD, MDD, and SCZ." NeurIPS 2026 (under review).
```

Citation will be updated after the review period.
