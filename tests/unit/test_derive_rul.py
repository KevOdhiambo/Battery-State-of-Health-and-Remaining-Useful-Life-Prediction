"""Tests for RUL crossing, censoring, and baseline math."""

import numpy as np
import pandas as pd
import pytest

from src.models.derive_rul import crossing_from_trajectory, slope_rul, true_eol_cycle


class TestTrueEol:
    def test_first_crossing_cycle(self):
        df = pd.DataFrame({"cycle_index": [1, 2, 3, 4], "soh": [0.8, 0.72, 0.69, 0.71]})
        assert true_eol_cycle(df, 0.70) == 3

    def test_censored_battery_returns_none(self):
        df = pd.DataFrame({"cycle_index": [1, 2], "soh": [0.9, 0.85]})
        assert true_eol_cycle(df, 0.70) is None


class TestTrajectoryCrossing:
    def test_interpolates_between_grid_points(self):
        horizons = np.array([1.0, 6.0, 11.0])
        soh = np.array([0.75, 0.71, 0.67])
        # Crossing of 0.70 lies between 6 and 11: 6 + (0.71-0.70)/(0.71-0.67)*5 = 7.25
        assert crossing_from_trajectory(horizons, soh, 0.70) == pytest.approx(7.25)

    def test_already_below_at_first_horizon(self):
        assert crossing_from_trajectory(
            np.array([1.0, 6.0]), np.array([0.65, 0.6]), 0.70
        ) == pytest.approx(1.0)

    def test_extrapolates_last_segment(self):
        horizons = np.array([1.0, 6.0])
        soh = np.array([0.80, 0.75])  # slope -0.01/cycle, reaches 0.70 five cycles past grid
        assert crossing_from_trajectory(horizons, soh, 0.70) == pytest.approx(11.0)

    def test_non_degrading_trajectory_is_censored(self):
        assert crossing_from_trajectory(
            np.array([1.0, 6.0]), np.array([0.80, 0.80]), 0.70
        ) is None


class TestSlopeRul:
    def test_closed_form_crossing(self):
        assert slope_rul(0.80, -0.002, 0.70) == pytest.approx(50.0)

    def test_already_below_threshold(self):
        assert slope_rul(0.69, -0.002, 0.70) == 0.0

    def test_non_negative_slope_censored(self):
        assert slope_rul(0.80, 0.0, 0.70) is None
