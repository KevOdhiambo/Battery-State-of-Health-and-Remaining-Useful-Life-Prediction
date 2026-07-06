"""Unit-level train/val/test splits: whole batteries, never cycles.

Random cycle-level splits leak catastrophically on this data -- the model
interpolates between adjacent cycles of the same cell and posts a fake
high score. Every split boundary here is a battery id.

Cohort rationale (from the NASA experiment READMEs and parse-time audit):

- B0005/B0006/B0007/B0018: 24 C, consistent 2 A protocol, full fade
  curves to ~30 percent fade. These are the only batteries whose capacity
  trace is a clean SoH signal end to end, so validation and test are
  drawn exclusively from them.
- B0025-B0028 (24 C, short runs) and B0029-B0032 (43 C, mild fade):
  consistent protocol per battery, so they are usable as extra TRAINING
  material, but their short/mild degradation makes them weak evaluation
  targets.
- Everything else is excluded: mixed per-cycle loads and cutoff voltages
  make measured capacity incomparable across cycles (B0033-B0044), NASA
  itself flags unexplained low-capacity runs and a control-software crash
  (B0041-B0052), and 4 C operation depresses capacity below the 70
  percent EOL threshold from the first cycle, which breaks the SoH-vs-
  rated definition and RUL outright (B0041-B0056).

Test battery choice: B0018 has the shortest, noisiest canonical curve
(132 cycles, no impedance interruptions pattern of the others) -- holding
it out makes the test harder, not easier. B0007 is validation.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_BATTERIES: tuple[str, ...] = (
    "B0005",
    "B0006",
    "B0025",
    "B0026",
    "B0027",
    "B0028",
    "B0029",
    "B0030",
    "B0031",
    "B0032",
)
VAL_BATTERIES: tuple[str, ...] = ("B0007",)
TEST_BATTERIES: tuple[str, ...] = ("B0018",)

EXCLUDED_BATTERIES: dict[str, str] = {
    "B0033": (
        "4 A discharge to 2.0 V; capacity not comparable to the 2 A cohort, "
        "anomalous near-zero first cycles"
    ),
    "B0034": "4 A discharge to 2.2 V; capacity not comparable to the 2 A cohort",
    "B0036": "capacity readings exceed rated by 22 percent; measurement suspect",
    "B0038": (
        "load current and ambient temperature change across cycles; "
        "capacity trace is not an SoH signal"
    ),
    "B0039": "load current and ambient temperature change across cycles",
    "B0040": "load current and ambient temperature change across cycles",
    "B0041": "4 C ambient with mixed 4 A / 1 A loads; NASA-flagged unexplained low-capacity runs",
    "B0042": "4 C ambient, mixed loads; NASA-flagged unexplained low-capacity runs",
    "B0043": "4 C ambient, mixed loads; NASA-flagged unexplained low-capacity runs",
    "B0044": "4 C ambient, mixed loads; NASA-flagged unexplained low-capacity runs",
    "B0045": (
        "4 C ambient; capacity sits below the rated-capacity EOL threshold from the first cycle"
    ),
    "B0046": "4 C ambient; below-threshold from first cycle",
    "B0047": "4 C ambient; below-threshold from first cycle",
    "B0048": "4 C ambient; below-threshold from first cycle",
    "B0049": (
        "experiment ended by control-software crash; SoH values up to 1.19 "
        "are physically impossible"
    ),
    "B0050": "software-crash batch; SoH values up to 1.32, 4 of 25 discharges lack capacity",
    "B0051": "software-crash batch; SoH values up to 1.17",
    "B0052": "software-crash batch; only 4 usable discharge cycles",
    "B0053": "4 C ambient; below-threshold from first cycle",
    "B0054": "4 C ambient; below-threshold from first cycle",
    "B0055": "4 C ambient; below-threshold from first cycle",
    "B0056": "4 C ambient; below-threshold from first cycle",
}

SPLIT_OF: dict[str, str] = (
    dict.fromkeys(TRAIN_BATTERIES, "train")
    | dict.fromkeys(VAL_BATTERIES, "val")
    | dict.fromkeys(TEST_BATTERIES, "test")
)


class SplitLeakageError(AssertionError):
    """A battery id appears in more than one split. Everything downstream is invalid."""


def assert_no_battery_overlap(split_of_battery: pd.Series) -> None:
    """Fail loudly if any battery id maps to more than one split.

    Args:
        split_of_battery: Series indexed or valued so that grouping battery_id
            against split labels is possible; expects a frame-like input with
            one split label per row, indexed by battery_id.

    Raises:
        SplitLeakageError: If any battery id carries two different split labels.
    """
    n_splits_per_battery = split_of_battery.groupby(level=0).nunique()
    leaked = n_splits_per_battery[n_splits_per_battery > 1]
    if not leaked.empty:
        raise SplitLeakageError(
            f"battery ids present in multiple splits: {sorted(leaked.index)} -- "
            "unit-level isolation is broken and all downstream results are invalid"
        )


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a split column and drop excluded batteries.

    Args:
        df: Feature or cycle table with a battery_id column.

    Returns:
        Copy restricted to cohort batteries with a split column in
        {train, val, test}.

    Raises:
        KeyError: If a battery id is neither in the cohort nor in the
            documented exclusion list -- an undocumented battery must be a
            deliberate decision, not a silent default.
        SplitLeakageError: If the resulting assignment maps any battery to
            more than one split.
    """
    unknown = set(df["battery_id"]) - set(SPLIT_OF) - set(EXCLUDED_BATTERIES)
    if unknown:
        raise KeyError(
            f"battery ids with no split assignment and no documented exclusion: {sorted(unknown)}"
        )
    excluded_present = sorted(set(df["battery_id"]) & set(EXCLUDED_BATTERIES))
    if excluded_present:
        logger.info("dropping %d excluded batteries: %s", len(excluded_present), excluded_present)
    out = df[df["battery_id"].isin(SPLIT_OF)].copy()
    out["split"] = out["battery_id"].map(SPLIT_OF)
    assert_no_battery_overlap(out.set_index("battery_id")["split"])
    for name in ("train", "val", "test"):
        part = out[out["split"] == name]
        logger.info(
            "%s: %d rows, batteries %s", name, len(part), sorted(part["battery_id"].unique())
        )
    return out


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {train, val, test} frames from assign_splits output."""
    assigned = assign_splits(df)
    return {name: assigned[assigned["split"] == name].copy() for name in ("train", "val", "test")}
