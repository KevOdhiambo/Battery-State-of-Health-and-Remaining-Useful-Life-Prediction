"""Unit tests for the NASA .mat parser's pure logic.

Synthetic structs mimic scipy.io.loadmat output: object-dtype record
arrays of shape (1, 1) whose fields hold (1, n) float arrays.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.parse_nasa import (
    DataValidationError,
    matlab_datevec_to_datetime,
    summarize_charge,
    summarize_discharge,
    validate_cycles_table,
)


def make_data_struct(fields: dict[str, np.ndarray]) -> np.ndarray:
    arr = np.zeros((1, 1), dtype=[(name, "O") for name in fields])
    for name, values in fields.items():
        arr[name][0, 0] = np.asarray(values, dtype=float).reshape(1, -1)
    return arr


def valid_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery_id": ["B0005", "B0005"],
            "cycle_index": [1, 2],
            "capacity_ah": [1.85, 1.84],
            "discharge_duration_s": [3600.0, 3590.0],
        }
    )


class TestMatlabDatevec:
    def test_converts_fractional_seconds(self):
        vec = np.array([[2008.0, 4.0, 2.0, 13.0, 8.0, 17.5]])
        assert matlab_datevec_to_datetime(vec) == datetime(2008, 4, 2, 13, 8, 17, 500000)

    def test_rejects_short_vector(self):
        with pytest.raises(ValueError, match="6-element"):
            matlab_datevec_to_datetime(np.array([2008.0, 4.0, 2.0]))


class TestSummaries:
    def test_discharge_summary(self):
        data = make_data_struct(
            {
                "Voltage_measured": np.array([4.2, 3.5, 2.7]),
                "Current_measured": np.array([-2.0, -2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 30.0, 28.0]),
                "Time": np.array([0.0, 1800.0, 3600.0]),
            }
        )
        s = summarize_discharge(data)
        assert s["discharge_duration_s"] == 3600.0
        assert s["voltage_min"] == 2.7
        assert s["temp_max"] == 30.0
        assert s["current_mean"] == -2.0

    def test_charge_summary_taper_current(self):
        data = make_data_struct(
            {
                "Current_measured": np.array([1.5, 1.5, 0.02]),
                "Temperature_measured": np.array([24.0, 26.0, 25.0]),
                "Time": np.array([0.0, 5000.0, 10000.0]),
            }
        )
        s = summarize_charge(data)
        assert s["charge_duration_s"] == 10000.0
        assert s["charge_current_end"] == 0.02
        assert s["charge_temp_max"] == 26.0


class TestValidation:
    def test_accepts_valid_table(self):
        validate_cycles_table(valid_table())

    def test_rejects_empty(self):
        with pytest.raises(DataValidationError, match="empty"):
            validate_cycles_table(valid_table().iloc[0:0])

    def test_rejects_out_of_range_capacity(self):
        df = valid_table()
        df.loc[0, "capacity_ah"] = 9.9
        with pytest.raises(DataValidationError, match="capacity outside"):
            validate_cycles_table(df)

    def test_rejects_duplicate_cycle(self):
        df = valid_table()
        df.loc[1, "cycle_index"] = 1
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_cycles_table(df)

    def test_rejects_null_capacity(self):
        df = valid_table()
        df.loc[0, "capacity_ah"] = np.nan
        with pytest.raises(DataValidationError, match="null"):
            validate_cycles_table(df)
