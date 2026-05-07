# EEG Evaluation Traps

Code and data for:
**"Evaluation Traps in EEG Disease Classification: Identity Leakage, Lucky Folds, and Objective Mismatch — Replicated Across AD, FTD, MDD, and SCZ"**

Submitted to NeurIPS 2026 Evaluations & Datasets Track.

---

## Status: Coming Soon

This repository is being cleaned up for public release. Split manifests, pipeline configurations, YAML configs, and feature extraction code will be documented and released here.

If you need something urgently, open an issue and I will prioritize it. : ) .

**Note on anonymization:** This repository is being carefully reviewed to remove identifying information. This is non-trivial and we want to be really careful not to let something slip. The pipeline runs on HPC infrastructure with Docker containers, and configuration files may contain paths or environment variables tied to the original compute environment. If you spot anything that should be removed, please open an issue.

---

## What will be here

- `manifests/` — split manifests (held-out subject IDs per fold for all 5 cohorts)
- `configs/` — YAML pipeline configurations
- `features/` — feature extraction code
- `evaluation/` — evaluation loop and metrics code
- `bids_builders/` — BIDS curation scripts for MDD and SCZ datasets

---

## Datasets

- AD / FTD: OpenNeuro ds004504 / ds004904
- MDD: Mumtaz et al. 2017 (figshare DOI 10.6084/m9.figshare.4244171)
- SCZ: Olejarczyk & Jernajczyk 2017 (RepOD DOI 10.18150/repod.0107441)
