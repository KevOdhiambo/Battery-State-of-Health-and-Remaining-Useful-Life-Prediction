"""Tests for unit-level split assignment and the leakage assertion."""

import pandas as pd
import pytest

from src.data.splits import (
    EXCLUDED_BATTERIES,
    SPLIT_OF,
    SplitLeakageError,
    assert_no_battery_overlap,
    assign_splits,
    split_frames,
)


def cohort_df() -> pd.DataFrame:
    rows = [(b, i) for b in SPLIT_OF for i in (1, 2, 3)]
    return pd.DataFrame(rows, columns=["battery_id", "cycle_index"])


def test_every_battery_in_exactly_one_split():
    ids = list(SPLIT_OF)
    assert len(ids) == len(set(ids))
    assert set(SPLIT_OF.values()) == {"train", "val", "test"}


def test_no_overlap_between_cohort_and_exclusions():
    assert not set(SPLIT_OF) & set(EXCLUDED_BATTERIES)


def test_assertion_fails_loudly_on_overlap():
    s = pd.Series(
        ["train", "test"], index=pd.Index(["B0005", "B0005"], name="battery_id"), name="split"
    )
    with pytest.raises(SplitLeakageError, match="B0005"):
        assert_no_battery_overlap(s)


def test_assertion_passes_on_clean_assignment():
    assigned = assign_splits(cohort_df())
    assert_no_battery_overlap(assigned.set_index("battery_id")["split"])


def test_excluded_batteries_are_dropped():
    df = pd.concat(
        [cohort_df(), pd.DataFrame({"battery_id": ["B0049"], "cycle_index": [1]})],
        ignore_index=True,
    )
    assigned = assign_splits(df)
    assert "B0049" not in set(assigned["battery_id"])


def test_unknown_battery_raises():
    df = pd.concat(
        [cohort_df(), pd.DataFrame({"battery_id": ["B9999"], "cycle_index": [1]})],
        ignore_index=True,
    )
    with pytest.raises(KeyError, match="B9999"):
        assign_splits(df)


def test_split_frames_partition_rows():
    frames = split_frames(cohort_df())
    total = sum(len(f) for f in frames.values())
    assert total == len(cohort_df())
    assert set(frames["test"]["battery_id"]) == {"B0018"}
    assert set(frames["val"]["battery_id"]) == {"B0007"}
