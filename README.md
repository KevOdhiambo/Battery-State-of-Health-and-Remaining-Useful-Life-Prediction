# Battery State-of-Health & Remaining-Useful-Life Prediction

Predicts lithium-ion battery State-of-Health (SoH) from charge/discharge cycle data
on the NASA PCoE Li-ion Battery Aging dataset, with Remaining-Useful-Life (RUL)
derived as a separately reported extension.

**Status: work in progress — Stage 1 (data acquisition) underway.**

Results, baseline comparison, and limitations will be added here once the pipeline
is built and verified. This README will lead with honest per-battery numbers next
to a naive baseline, not a headline score.

## Design principles

1. **Split by battery unit, never by cycle.** Validation/test batteries are entirely
   unseen during training. Cycle-level random splits leak catastrophically.
2. **SoH is the primary target; RUL is derived and reported separately** as the
   harder, higher-variance task.
3. **Naive baseline first.** The model only claims value if it beats it on held-out
   batteries.
4. **No target leakage.** Every feature is computable from the current and past
   cycles only.

## Data

NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging dataset,
downloaded from the NASA PHM datasets S3 mirror
(`https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip`).
Raw `.mat` files live under `data/raw/` and are not committed.

## Layout

```
src/
  data/          # NASA .mat parsing, unit-level splits
  features/      # leakage-safe per-cycle feature engineering
  models/        # SoH training, RUL derivation
  eval/          # per-battery reports and plots
  monitoring/    # PSI-style input drift checks
tests/           # unit tests, incl. leakage assertions
data/raw/        # raw NASA files (gitignored)
data/processed/  # parsed per-cycle tables (gitignored)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
