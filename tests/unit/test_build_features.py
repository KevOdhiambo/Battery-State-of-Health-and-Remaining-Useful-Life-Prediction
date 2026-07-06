"""Leakage and correctness tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    RATED_CAPACITY_AH,
    TARGET_COLUMN,
    build_features,
)


def make_cycles(n: int = 30, battery_id: str = "B0005", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "battery_id": battery_id,
            "cycle_index": np.arange(1, n + 1),
            "cycle_start_time": pd.date_range("2008-04-02", periods=n, freq="6h"),
            "ambient_temperature": 24.0,
            "capacity_ah": np.linspace(1.86, 1.30, n) + rng.normal(0, 0.005, n),
            "discharge_duration_s": np.linspace(3700, 2800, n),
            "voltage_min": rng.uniform(2.5, 2.7, n),
            "voltage_mean": 3.3,
            "current_mean": -2.0,
            "temp_max": rng.uniform(38, 41, n),
            "temp_mean": 33.0,
            "charge_duration_s": np.linspace(10000, 10800, n),
            "charge_current_end": rng.normal(0, 0.003, n),
            "charge_temp_max": 30.0,
            "charge_temp_mean": 27.0,
        }
    )


def test_prefix_invariance_no_future_information():
    """Features for cycles <= t must not change when later cycles are removed.

    This is the mechanical definition of leakage-safety: if truncating the
    future changes a feature value in the past, that feature saw the future.
    """
    full = build_features(make_cycles(n=30))
    truncated = build_features(make_cycles(n=30).iloc[:15])
    pd.testing.assert_frame_equal(
        full.iloc[:15].reset_index(drop=True)[FEATURE_COLUMNS],
        truncated.reset_index(drop=True)[FEATURE_COLUMNS],
    )


def test_soh_definition():
    df = build_features(make_cycles())
    expected = df["capacity_ah"] / RATED_CAPACITY_AH
    pd.testing.assert_series_equal(df[TARGET_COLUMN], expected, check_names=False)


def test_soh_prev_is_shifted_not_current():
    df = build_features(make_cycles())
    assert np.isnan(df["soh_prev"].iloc[0])
    np.testing.assert_allclose(
        df["soh_prev"].iloc[1:].to_numpy(), df[TARGET_COLUMN].iloc[:-1].to_numpy()
    )


def test_rolling_features_exclude_current_cycle():
    df = build_features(make_cycles())
    # At row 6 (0-based), the 5-window shifted mean covers rows 1..5 only.
    expected = df[TARGET_COLUMN].iloc[1:6].mean()
    assert df["soh_roll_mean"].iloc[6] == pytest.approx(expected)


def test_lag_features_do_not_cross_batteries():
    two = pd.concat(
        [make_cycles(n=10, battery_id="B0005"), make_cycles(n=10, battery_id="B0006", seed=8)],
        ignore_index=True,
    )
    df = build_features(two)
    first_b6 = df[df["battery_id"] == "B0006"].iloc[0]
    assert np.isnan(first_b6["soh_prev"])
    assert np.isnan(first_b6["time_since_prev_cycle_h"])


def test_flagged_features_stay_excluded():
    flagged = {name for name, (_, f) in FEATURE_SPECS.items() if f}
    assert flagged == {"discharge_duration_s", "voltage_min", "temp_max"}
    assert not flagged & set(FEATURE_COLUMNS)


def test_every_feature_column_has_justification():
    assert set(FEATURE_COLUMNS) <= set(FEATURE_SPECS)
