"""Leakage-safe per-cycle features for SoH prediction.

Prediction setup: estimate SoH of discharge cycle t for a battery given
(a) the cycle position and operating conditions known before the discharge
starts, (b) telemetry from the charge phase that precedes the discharge,
and (c) anything observed on PAST cycles (t-1 and earlier), including past
measured SoH -- the naive baseline "predict last observed SoH" already
assumes past SoH is known, so the model gets the same information.

What is deliberately NOT a feature: any measurement taken during the
discharge being predicted. At constant discharge current, duration to the
voltage cutoff is coulomb counting -- discharge_duration_s * current IS
the capacity, i.e. the label. Including it turns the task into reading
the answer off the sensor. Same reasoning excludes the current cycle's
voltage_min (it is the cutoff condition) and its temperature profile
(recorded during the discharge). Their PAST-cycle values are fine.

Every feature is justified in FEATURE_SPECS; anything borderline is
flagged there instead of silently included.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RATED_CAPACITY_AH = 2.0
EOL_SOH = 0.7  # NASA ran these experiments to 30 percent fade from rated
ROLLING_WINDOW = 5
SLOPE_WINDOW = 5

# feature name -> (justification, borderline flag). Borderline features are
# excluded from FEATURE_COLUMNS until explicitly approved.
FEATURE_SPECS: dict[str, tuple[str, bool]] = {
    "cycle_index": (
        "Ordinal position of the discharge cycle; known before the cycle runs.",
        False,
    ),
    "ambient_temperature": (
        "Chamber setpoint for the cycle; an operating condition set before the cycle runs.",
        False,
    ),
    "time_since_prev_cycle_h": (
        "Gap between this cycle's start and the previous cycle's start; both timestamps are "
        "known when the cycle begins. Captures rest-driven capacity regeneration.",
        False,
    ),
    "charge_duration_s": (
        "Duration of the charge phase that PRECEDES this discharge; fully observed before "
        "the discharge starts. Degraded cells spend longer in CV, so this is genuine "
        "telemetry signal, not a re-encoding of the discharge measurement.",
        False,
    ),
    "charge_current_end": (
        "CV taper current at charge cutoff, from the preceding charge phase; observed "
        "before the discharge starts.",
        False,
    ),
    "charge_temp_max": (
        "Peak cell temperature during the preceding charge; observed before the discharge.",
        False,
    ),
    "soh_prev": (
        "Last observed SoH (shift 1). Past measurement; identical information to the "
        "naive last-value baseline.",
        False,
    ),
    "soh_roll_mean": (
        f"Mean of the previous {ROLLING_WINDOW} observed SoH values, shifted so the "
        "current cycle is excluded.",
        False,
    ),
    "soh_roll_std": (
        f"Std of the previous {ROLLING_WINDOW} observed SoH values, shifted; captures "
        "local noise / regeneration activity.",
        False,
    ),
    "fade_rate": (
        f"Per-cycle SoH slope over the previous {SLOPE_WINDOW} observed values "
        "(shifted); the local degradation rate.",
        False,
    ),
    "discharge_duration_prev": (
        "PREVIOUS cycle's discharge duration (shift 1); a past observation.",
        False,
    ),
    "voltage_min_prev": (
        "PREVIOUS cycle's minimum discharge voltage (shift 1); a past observation.",
        False,
    ),
    "temp_max_prev": (
        "PREVIOUS cycle's peak discharge temperature (shift 1); a past observation.",
        False,
    ),
    "discharge_duration_s": (
        "BORDERLINE, EXCLUDED: duration of the discharge being predicted. At constant "
        "current this is coulomb counting -- duration * current = capacity = the label.",
        True,
    ),
    "voltage_min": (
        "BORDERLINE, EXCLUDED: minimum voltage of the discharge being predicted; it is "
        "the cutoff condition of the measurement that defines the label.",
        True,
    ),
    "temp_max": (
        "BORDERLINE, EXCLUDED: peak temperature during the discharge being predicted; "
        "recorded while the label is being measured.",
        True,
    ),
}

FEATURE_COLUMNS: list[str] = [name for name, (_, flagged) in FEATURE_SPECS.items() if not flagged]
TARGET_COLUMN = "soh"


def _slope(values: np.ndarray) -> float:
    """OLS slope of a short window of SoH values against cycle offsets."""
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def build_features(cycles: pd.DataFrame) -> pd.DataFrame:
    """Add the SoH target and leakage-safe features to the per-cycle table.

    All rolling/lag features are computed per battery on the frame sorted by
    cycle_index, with shift(1) applied BEFORE any window so the current
    cycle's own measurement never enters its features. Early cycles get NaN
    lag features rather than an imputed value -- the model layer decides how
    to handle them, and silently backfilling here would leak cycle t into
    cycle t's own features.

    Args:
        cycles: Parsed per-cycle table from src.data.parse_nasa.

    Returns:
        Copy of the input, sorted by (battery_id, cycle_index), with the
        target column and every column in FEATURE_COLUMNS added.
    """
    df = cycles.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True).copy()
    df[TARGET_COLUMN] = df["capacity_ah"] / RATED_CAPACITY_AH

    grouped = df.groupby("battery_id", sort=False)
    soh_shifted = grouped[TARGET_COLUMN].shift(1)
    df["soh_prev"] = soh_shifted
    df["soh_roll_mean"] = (
        grouped[TARGET_COLUMN]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean())
    )
    df["soh_roll_std"] = (
        grouped[TARGET_COLUMN]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=2).std())
    )
    df["fade_rate"] = (
        grouped[TARGET_COLUMN]
        .transform(
            lambda s: s.shift(1).rolling(SLOPE_WINDOW).apply(_slope, raw=True)
        )
    )
    df["discharge_duration_prev"] = grouped["discharge_duration_s"].shift(1)
    df["voltage_min_prev"] = grouped["voltage_min"].shift(1)
    df["temp_max_prev"] = grouped["temp_max"].shift(1)
    df["time_since_prev_cycle_h"] = (
        grouped["cycle_start_time"].diff().dt.total_seconds() / 3600.0
    )

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"feature build produced no column(s): {missing}")
    logger.info(
        "built %d features for %d rows across %d batteries",
        len(FEATURE_COLUMNS), len(df), df["battery_id"].nunique(),
    )
    return df


def feature_justification_table() -> pd.DataFrame:
    """Feature -> justification -> included/flagged, for review and the README."""
    return pd.DataFrame(
        [
            {
                "feature": name,
                "included": not flagged,
                "justification": text,
            }
            for name, (text, flagged) in FEATURE_SPECS.items()
        ]
    )


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build leakage-safe features")
    parser.add_argument("--cycles", type=Path, default=Path("data/processed/cycles.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/features.parquet"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = build_features(pd.read_parquet(args.cycles))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
