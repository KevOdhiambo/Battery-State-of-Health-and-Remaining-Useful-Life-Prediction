"""Tests for horizon target construction and baseline math."""

import numpy as np
import pandas as pd

from src.models.train_soh import (
    add_horizon_targets,
    baseline_global_slope,
    evaluable,
    global_fade_slope,
    horizon_target,
)


def toy(n: int = 8, battery_id: str = "B1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery_id": battery_id,
            "cycle_index": np.arange(1, n + 1),
            "soh": np.round(1.0 - 0.01 * np.arange(n), 4),
            "soh_prev": [np.nan] + list(np.round(1.0 - 0.01 * np.arange(n - 1), 4)),
        }
    )


def test_horizon_1_is_current_cycle_soh():
    df = add_horizon_targets(toy(), (1,))
    pd.testing.assert_series_equal(df[horizon_target(1)], df["soh"], check_names=False)


def test_horizon_k_pulls_soh_k_minus_1_ahead():
    df = add_horizon_targets(toy(n=8), (3,))
    # Row t targets soh(t + 2): row 0 -> soh of row 2.
    assert df[horizon_target(3)].iloc[0] == df["soh"].iloc[2]
    # Last k-1 rows have no target.
    assert df[horizon_target(3)].iloc[-2:].isna().all()


def test_horizon_targets_do_not_cross_batteries():
    two = pd.concat([toy(battery_id="B1"), toy(battery_id="B2")], ignore_index=True)
    df = add_horizon_targets(two, (3,))
    b1 = df[df["battery_id"] == "B1"]
    assert b1[horizon_target(3)].iloc[-2:].isna().all()


def test_evaluable_requires_baseline_and_label():
    df = add_horizon_targets(toy(n=8), (3,))
    ev = evaluable(df, 3)
    # Drops the first row (no soh_prev) and the last two (no horizon-3 label).
    assert len(ev) == 5


def test_global_slope_baseline_steps_k_times():
    df = add_horizon_targets(toy(), (1, 5))
    train = df.rename(columns={})
    slope = global_fade_slope(train)
    assert np.isclose(slope, -0.01)
    ev = evaluable(df, 5)
    pred = baseline_global_slope(ev, slope, 5)
    np.testing.assert_allclose(pred, ev["soh_prev"].to_numpy() - 0.05)
