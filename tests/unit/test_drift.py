"""Tests for the PSI drift check."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    drift_report,
    population_stability_index,
    rate_psi,
)


def test_identical_distributions_are_stable():
    rng = np.random.default_rng(0)
    sample = pd.Series(rng.normal(0, 1, 5000))
    assert population_stability_index(sample, sample) < 0.01


def test_shifted_distribution_is_flagged():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(2, 1, 5000))
    assert population_stability_index(ref, shifted) > 0.25


def test_out_of_range_values_are_caught_by_open_edges():
    ref = pd.Series(np.linspace(0, 1, 1000))
    beyond = pd.Series(np.full(500, 5.0))
    assert population_stability_index(ref, beyond) > 0.25


def test_constant_reference_feature_does_not_crash():
    ref = pd.Series(np.full(100, 3.0))
    same = pd.Series(np.full(100, 3.0))
    moved = pd.Series(np.full(100, 4.0))
    assert population_stability_index(ref, same) < 0.01
    assert population_stability_index(ref, moved) > 0.25


def test_empty_sample_raises():
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(pd.Series([], dtype=float), pd.Series([1.0]))


def test_ratings():
    assert rate_psi(0.05) == "stable"
    assert rate_psi(0.15) == "moderate shift"
    assert rate_psi(0.30) == "significant shift"


def test_drift_report_sorted_worst_first():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(3, 1, 2000)})
    report = drift_report(ref, cur, ["a", "b"])
    assert list(report["feature"]) == ["b", "a"]
    assert report.loc[0, "rating"] == "significant shift"
