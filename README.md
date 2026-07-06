# Battery State-of-Health & Remaining-Useful-Life Prediction

End-to-end ML pipeline predicting lithium-ion battery State-of-Health (SoH) from
charge/discharge cycle data on the NASA PCoE Li-ion Battery Aging dataset, with
Remaining-Useful-Life (RUL) derived as a separately reported extension.

Priority is defensibility over headline metrics: whole batteries are held out
(never cycles), the naive baseline is reported before the model, and results
that look too good are treated as bugs until audited.

## Results (honest numbers, baseline first)

Held-out batteries: **val = B0007, test = B0018** -- whole batteries the models
never saw. SoH = measured capacity / 2.0 Ah rated. All errors in SoH units
(0.01 = 1% of rated capacity).

### SoH prediction error (MAE) by forecast horizon

| Horizon (cycles ahead) | Method | val B0007 | test B0018 |
|---|---|---|---|
| 1 | baseline: last SoH + global slope | **0.0031** | **0.0061** |
| 1 | LightGBM | 0.0051 | 0.0088 |
| 10 | baseline: last SoH + global slope | **0.0089** | **0.0178** |
| 10 | LightGBM | 0.0171 | 0.0197 |
| 25 | baseline: last SoH + global slope | **0.0142** | 0.0259 |
| 25 | LightGBM | 0.0326 | **0.0192** |
| 50 | baseline: last SoH + global slope | **0.0189** | 0.0295 |
| 50 | LightGBM | 0.0497 | **0.0126** |

Read the table honestly:

- **At horizons 1-10 the naive baseline wins everywhere.** SoH moves ~0.003 per
  cycle; when the previous cycle's measured SoH is known, "last value + average
  fade" is near-optimal and the model adds nothing. Any project reporting a
  model win at horizon 1 without this baseline should be read skeptically.
- **At horizons 25-50 the result is split across the two held-out batteries.**
  The model beats the baseline convincingly on B0018 (noisy, regeneration-heavy
  fade -- learned dynamics help) and loses on B0007 (shallow near-linear fade --
  a constant slope is ideal). With a two-battery evaluation universe this split
  IS the result; an aggregate would hide it.

### RUL error (cycles), derived from projected SoH trajectories

RUL = cycles until SoH first crosses end-of-life. Primary EOL = 70% of rated
capacity (NASA ran the canonical cells to 30% fade); 80% reported as
sensitivity. Evaluated at every standing point from cycle 10 onward.

| Battery | EOL | Method | RUL MAE | Median AE | Max AE |
|---|---|---|---|---|---|
| B0018 (test) | 70% | baseline: global slope | 19.8 | 18.3 | 52.2 |
| B0018 (test) | 70% | **LightGBM trajectory** | **10.0** | 9.8 | 18.1 |
| B0018 (test) | 80% | baseline: global slope | 16.4 | 16.5 | 35.5 |
| B0018 (test) | 80% | **LightGBM trajectory** | **9.8** | 8.7 | 22.7 |
| B0007 (val) | 70% | -- | right-censored: B0007 never reaches 70% in its recorded life (min SoH 0.7002) | | |
| B0007 (val) | 80% | baseline: global slope | 15.6 | 17.3 | 34.6 |
| B0007 (val) | 80% | LightGBM trajectory | **13.5** | 14.2 | 22.9 |

**SoH MAE is ~0.006-0.03 depending on horizon; RUL error is ~10-20 cycles. RUL
is the harder task and its error is correspondingly larger**: crossing-point
geometry amplifies trajectory error (at the observed fade rate of ~0.0016
SoH/cycle, a 0.01 SoH error moves the predicted crossing by ~6 cycles), and the
answer is threshold-sensitive -- at 70% the validation battery has no true EOL
at all.

Full per-battery tables, projection plots, and the drift check:
[reports/results.md](reports/results.md).

## Leakage audit

- **Unit-level split.** Batteries, not cycles, are assigned to splits
  ([src/data/splits.py](src/data/splits.py)). A random cycle split would let the
  model interpolate between adjacent cycles of the same cell and post fake
  scores. A `SplitLeakageError` assertion fails the pipeline if any battery id
  appears in more than one split, and a unit test proves it fires.
- **No future information in features.** Every feature is computable from the
  current cycle's pre-discharge information plus past cycles
  ([src/features/build_features.py](src/features/build_features.py) documents a
  justification per feature). A prefix-invariance test rebuilds features on
  truncated history and asserts bit-identical values -- if any feature saw the
  future, that test fails.
- **Label-adjacent telemetry excluded.** The predicted cycle's own discharge
  duration, minimum voltage, and peak temperature are flagged and excluded: at
  constant current, discharge duration IS coulomb counting, i.e. the label.
- **Too-good tripwire.** R^2 > 0.95 on unseen batteries is logged as a leakage
  suspect. It fired at horizon 1; the audit showed the baselines score even
  higher R^2 (0.978-0.994) -- saturation from target autocorrelation, not
  leakage. R^2 is reported as a tripwire only, never as the headline.

## Limitations

- **Two evaluation batteries.** NASA's canonical, protocol-consistent cohort is
  four cells; two train, one validates, one tests. Every number above is "how
  well does this transfer to ONE unseen cell" -- noisy by construction. The
  train/val/test-split results should be read as a case study, not a benchmark.
- **22 of 34 batteries excluded, deliberately.** Mixed per-cycle loads and
  cutoff voltages make measured capacity incomparable across cycles
  (B0033-B0044), NASA itself flags unexplained low-capacity runs and a
  control-software crash (B0041-B0052), and 4 C operation depresses capacity
  below the 70% threshold from the first cycle (B0041-B0056). Each exclusion is
  documented per battery in `src/data/splits.py`.
- **Single chemistry, lab conditions.** 18650 Li-ion cells cycled with full
  constant-current discharges in a chamber. Real EV duty cycles have partial
  discharges, varying loads, and no per-cycle capacity measurements; the
  autoregressive features would need re-thinking there (the telemetry-only
  model variant, reported in results.md, is the honest floor for that setting).
- **Validation battery is the friendly one.** Model selection (early stopping)
  happens on B0007, the shallowest fade curve of the four.
- **EOL threshold sensitivity.** 70% vs 80% changes both the RUL answer and
  whether the question is answerable (B0007 is censored at 70%).

## Data

NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging dataset,
downloaded from the NASA PHM datasets mirror:
`https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip` (~200 MB).
Extract under `data/raw/extracted/`. Raw `.mat` files are nested MATLAB structs;
parsing decisions (cycle types, aborted runs, duplicated files, zero-capacity
partial discharges) are documented in
[src/data/parse_nasa.py](src/data/parse_nasa.py).

## Reproduce

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.parse_nasa          # .mat -> data/processed/cycles.parquet
python -m src.features.build_features  # leakage-safe features -> features.parquet
python -m src.models.train_soh         # baselines + per-horizon models -> artifacts/
python -m src.models.derive_rul        # RUL from projected trajectories
python -m src.eval.report              # figures + reports/results.md

pytest tests/unit -q                   # 44 tests incl. leakage assertions
```

## Layout

```
src/
  data/parse_nasa.py       # nested .mat -> validated per-cycle table
  data/splits.py           # unit-level splits + SplitLeakageError assertion
  features/build_features.py  # leakage-safe features, justification per feature
  models/train_soh.py      # baselines FIRST, then ridge / LightGBM per horizon
  models/derive_rul.py     # EOL crossing from projected trajectories
  eval/report.py           # per-battery curves and tables
  monitoring/drift.py      # PSI drift check on input features
tests/unit/                # incl. prefix-invariance and split-leakage tests
reports/                   # results.md + figures
```
