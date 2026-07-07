# Battery State-of-Health and Remaining-Useful-Life Prediction

Production-style machine learning pipeline that estimates lithium-ion battery
State of Health (SoH) from charge/discharge cycle telemetry and derives
Remaining Useful Life (RUL) from the projected degradation trajectory. Built on
the NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging dataset.

The design priority is defensibility over headline metrics: evaluation holds
out whole batteries (never individual cycles), every model is reported against
a naive baseline, features carry a documented leakage justification, and
results that look too good are treated as defects until audited.

## Contents

- [Overview](#overview)
- [Results](#results)
- [Validation design and leakage controls](#validation-design-and-leakage-controls)
- [Getting started](#getting-started)
- [Pipeline](#pipeline)
- [Testing and code quality](#testing-and-code-quality)
- [Monitoring](#monitoring)
- [Interactive demo](#interactive-demo)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Data provenance](#data-provenance)
- [Acronyms and terms](#acronyms-and-terms)
- [License](#license)

## Overview

| | |
|---|---|
| **Primary task** | Multi-horizon SoH regression: predict SoH 1, 10, 25, and 50 cycles ahead of the last observed measurement |
| **Derived task** | RUL: project the predicted SoH trajectory to its end-of-life (EOL) crossing, reported separately |
| **Data** | 34 NASA PCoE batteries; 12 admitted to the modeling cohort after a documented data-quality audit |
| **Split** | By battery unit: 10 train, 1 validation (B0007), 1 test (B0018) |
| **Models** | Naive baselines (last value, global fade slope), ridge regression, LightGBM (one direct model per horizon) |
| **Stack** | Python 3.11, pandas, scikit-learn, LightGBM, SciPy, matplotlib, Streamlit; black + ruff + pytest |

## Results

All figures below are measured on whole held-out batteries. SoH is defined as
measured discharge capacity divided by the 2.0 Ah rated capacity; an error of
0.01 equals 1% of rated capacity.

### SoH prediction error (MAE) by forecast horizon

| Horizon (cycles ahead) | Method | val B0007 | test B0018 |
|---|---|---|---|
| 1 | Baseline: last SoH + global slope | **0.0031** | **0.0061** |
| 1 | LightGBM | 0.0051 | 0.0088 |
| 10 | Baseline: last SoH + global slope | **0.0089** | **0.0178** |
| 10 | LightGBM | 0.0171 | 0.0197 |
| 25 | Baseline: last SoH + global slope | **0.0142** | 0.0259 |
| 25 | LightGBM | 0.0326 | **0.0192** |
| 50 | Baseline: last SoH + global slope | **0.0189** | 0.0295 |
| 50 | LightGBM | 0.0497 | **0.0126** |

Interpretation:

- **At horizons 1-10 the naive baseline wins on both batteries.** SoH moves
  roughly 0.003 per cycle, so when the previous cycle's measurement is
  available, "last value plus average fade" is near-optimal and the model adds
  no value. This floor is reported deliberately: a short-horizon model win
  claimed without this baseline would not be credible.
- **At horizons 25-50 the outcome differs by battery.** The model outperforms
  the baseline decisively on B0018, whose fade is noisy and
  regeneration-heavy, and underperforms on B0007, whose fade is shallow and
  near-linear. With a two-battery evaluation universe, this per-battery split
  is the finding; an aggregate figure would conceal it.

### RUL error, derived from projected SoH trajectories

RUL is the number of cycles until SoH first crosses the EOL threshold. The
primary threshold is 70% of rated capacity, matching the fade level NASA ran
the canonical cells to; 80% is reported as a sensitivity check. Errors are
computed at every valid standing point from cycle 10 onward.

| Battery | EOL | Method | MAE (cycles) | Median AE | Max AE |
|---|---|---|---|---|---|
| B0018 (test) | 70% | Baseline: global slope | 19.8 | 18.3 | 52.2 |
| B0018 (test) | 70% | **LightGBM trajectory** | **10.0** | 9.8 | 18.1 |
| B0018 (test) | 80% | Baseline: global slope | 16.4 | 16.5 | 35.5 |
| B0018 (test) | 80% | **LightGBM trajectory** | **9.8** | 8.7 | 22.7 |
| B0007 (val) | 70% | Not evaluable | right-censored: B0007 never reaches 70% in its recorded life (minimum SoH 0.7002) | | |
| B0007 (val) | 80% | Baseline: global slope | 15.6 | 17.3 | 34.6 |
| B0007 (val) | 80% | LightGBM trajectory | **13.5** | 14.2 | 22.9 |

SoH MAE is 0.006-0.03 depending on horizon; RUL error is 10-20 cycles. **RUL is
the harder task and its error is correspondingly larger**: crossing-point
geometry amplifies trajectory error (at the observed fade rate of ~0.0016
SoH/cycle, a 0.01 SoH error moves the predicted crossing by roughly 6 cycles),
and the answer is threshold-sensitive -- at 70% the validation battery has no
true EOL at all.

Complete per-battery tables, projection plots, and the drift report are in
[reports/results.md](reports/results.md).

## Validation design and leakage controls

Leakage is the dominant failure mode in battery-degradation modeling; each
control below is enforced in code, not by convention.

1. **Unit-level split** ([src/data/splits.py](src/data/splits.py)). Whole
   batteries are assigned to train/validation/test. A random cycle-level split
   would let the model interpolate between adjacent cycles of the same cell
   and post inflated scores. A `SplitLeakageError` assertion fails the
   pipeline if any battery id appears in more than one split; a unit test
   proves the assertion fires.
2. **No future information in features**
   ([src/features/build_features.py](src/features/build_features.py)). Every
   feature is computable from the current cycle's pre-discharge information
   plus past cycles, and each carries a one-line leakage justification in a
   reviewable registry. A prefix-invariance test rebuilds features on
   truncated history and asserts bit-identical values.
3. **Label-adjacent telemetry excluded.** The predicted cycle's own discharge
   duration, minimum voltage, and peak temperature are flagged and excluded:
   at constant discharge current, duration to cutoff is coulomb counting --
   arithmetically the label itself.
4. **Too-good tripwire.** Any R^2 above 0.95 on unseen batteries is logged as
   a leakage suspect rather than celebrated. The tripwire fired at horizon 1;
   the audit showed the baselines score even higher R^2 (0.978-0.994), so the
   saturation comes from target autocorrelation, not leakage. R^2 is therefore
   reported only as a tripwire, never as a headline.

## Getting started

Prerequisites: Python 3.11+, ~1 GB disk for the raw dataset.

```
git clone https://github.com/KevOdhiambo/Battery-State-of-Health-and-Remaining-Useful-Life-Prediction.git
cd Battery-State-of-Health-and-Remaining-Useful-Life-Prediction
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Download the dataset (~200 MB) and extract it under `data/raw/extracted/`:

```
curl -L -o battery_data_set.zip "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
```

The archive contains six inner zips (one per experiment batch); extract them in
place so the `.mat` files sit under
`data/raw/extracted/5. Battery Data Set/<batch>/B00xx.mat`.

## Pipeline

Each stage is an idempotent module with a CLI entry point; each reads from and
writes to a defined location and can be inspected independently.

```
python -m src.data.parse_nasa          # raw .mat -> data/processed/cycles.parquet
python -m src.features.build_features  # leakage-safe features -> features.parquet
python -m src.models.train_soh         # baselines + per-horizon models -> artifacts/
python -m src.models.derive_rul        # RUL from projected trajectories -> artifacts/
python -m src.eval.report              # figures + reports/results.md
```

Model artifacts are versioned directories containing the serialized models,
the exact training configuration, and the evaluation metrics; an existing
version is never overwritten in place.

## Testing and code quality

```
pytest tests/unit -q     # 44 tests
black src tests demo     # formatting (line length 100)
ruff check src tests demo
```

The test suite covers the parser's struct handling and validation gates, the
leakage assertions (prefix invariance, split isolation, horizon-target shift
direction), the RUL crossing/censoring mathematics, and the PSI drift
computation including its degenerate cases. Formatting and linting are
configured in `pyproject.toml`.

## Monitoring

[src/monitoring/drift.py](src/monitoring/drift.py) implements a Population
Stability Index (PSI) check on input features, with bins fixed by the training
distribution's quantiles and open-ended outer edges to catch out-of-range
values. Thresholds follow standard practice: below 0.1 stable, 0.1-0.25
moderate shift, above 0.25 significant shift. The evaluation report runs the
check against each held-out battery to demonstrate the hook; in a deployment
it would run on a schedule against incoming telemetry.

## Interactive demo

```
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Select a held-out battery and a standing point; the app projects the SoH
trajectory from that point, derives RUL against the chosen EOL threshold (70%
or 80%, with censoring surfaced explicitly), and shows the PSI drift table for
the features observed so far. The demo imports the same pipeline modules used
in training -- there is no reimplemented inference path -- but nothing in
`src/` depends on it, and the repository stands alone if `demo/` is deleted.

## Project structure

```
src/
  data/parse_nasa.py            # nested .mat -> validated per-cycle table
  data/splits.py                # unit-level splits + SplitLeakageError assertion
  features/build_features.py    # leakage-safe features with justification registry
  models/train_soh.py           # baselines first, then ridge / LightGBM per horizon
  models/derive_rul.py          # EOL crossing from projected trajectories
  eval/report.py                # per-battery curves, tables, results.md
  monitoring/drift.py           # PSI drift check on input features
tests/unit/                     # 44 tests incl. leakage and censoring cases
notebooks/eda_data_quality.ipynb  # the anomaly audit behind the cohort decision
demo/                           # disposable Streamlit app
reports/                        # results.md + figures
artifacts/                      # versioned model artifacts (gitignored)
data/                           # raw + processed data (gitignored)
```

## Limitations

- **Two evaluation batteries.** NASA's protocol-consistent canonical cohort is
  four cells: two train, one validates, one tests. Every reported number
  measures transfer to a single unseen cell and is noisy by construction;
  results should be read as a case study, not a benchmark.
- **22 of 34 batteries excluded, deliberately.** Mixed per-cycle loads and
  cutoff voltages make measured capacity incomparable across cycles
  (B0033-B0044); NASA flags unexplained low-capacity runs and a
  control-software crash (B0041-B0052); and 4 C operation depresses capacity
  below the 70% threshold from the first cycle (B0041-B0056). Each exclusion
  is documented per battery in `src/data/splits.py`, and the full audit is
  reproducible in `notebooks/eda_data_quality.ipynb`.
- **Single chemistry, laboratory conditions.** 18650 Li-ion cells cycled with
  full constant-current discharges in a thermal chamber. Real EV duty cycles
  involve partial discharges, varying loads, and no per-cycle capacity
  measurement; the autoregressive features would require rework in that
  setting. The telemetry-only model variant reported in results.md is the
  honest floor for it.
- **Model selection happens on the friendliest battery, and results are
  sensitive to the val/test assignment.** Early stopping uses B0007, the
  shallowest fade curve of the four canonical cells, and final reporting rests
  on a single test battery (B0018). With a four-battery cohort, swapping which
  cell validates and which tests could move the reported numbers materially.
  A fuller evaluation would rotate assignments via leave-one-battery-out
  cross-validation; the current results should be read as a single-fold case
  study, not a cross-validated benchmark.
- **EOL threshold sensitivity.** Moving the threshold from 70% to 80% changes
  both the RUL answer and whether the question is answerable (B0007 is
  right-censored at 70%).

## Data provenance

NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging dataset,
published by B. Saha and K. Goebel (2007), NASA Ames Research Center.
Distributed via the NASA PHM datasets mirror:
`https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip`.

Raw files are nested MATLAB structs. All parsing decisions -- cycle-type
handling, aborted and zero-capacity runs, duplicated files across batches --
are documented where they are made, in
[src/data/parse_nasa.py](src/data/parse_nasa.py).

## Acronyms and terms

| Acronym | Meaning |
|---|---|
| SoH | State of Health -- measured discharge capacity divided by the 2.0 Ah rated capacity; 1.0 = new, degrades toward the EOL threshold |
| RUL | Remaining Useful Life -- number of discharge cycles until SoH first crosses the EOL threshold |
| EOL | End of Life -- the SoH threshold at which a battery is considered worn out (70% of rated capacity primary, 80% as sensitivity) |
| Ah | Ampere-hour -- unit of electric charge; battery capacity is measured in Ah |
| MAE | Mean Absolute Error -- average of the absolute prediction errors |
| AE | Absolute Error (median AE / max AE = median and worst single error) |
| RMSE | Root Mean Squared Error -- like MAE but penalizes large errors more |
| R^2 | Coefficient of determination -- fraction of target variance explained; used here only as a leakage tripwire, never as the headline |
| PSI | Population Stability Index -- drift metric comparing a feature's current distribution against its training distribution (< 0.1 stable, 0.1-0.25 moderate, > 0.25 significant) |
| ML | Machine Learning |
| NASA | National Aeronautics and Space Administration |
| PCoE | Prognostics Center of Excellence -- the NASA laboratory that published the battery aging dataset |
| PHM | Prognostics and Health Management -- the research field; also the name of NASA's dataset repository |
| EV | Electric Vehicle |
| CC / CV | Constant Current / Constant Voltage -- the two phases of the charging protocol used in the experiments |
| LightGBM | Light Gradient Boosting Machine -- the gradient-boosted tree library used for the models |
| val / test | Validation split (model selection) / test split (final reporting); each is one whole held-out battery |

## License

MIT -- see [LICENSE](LICENSE).
