"""Parse NASA PCoE Li-ion battery aging .mat files into a per-cycle table.

The raw files are nested MATLAB structs: one struct per battery holding a
1xN array of cycle entries. Each entry has a type (charge / discharge /
impedance), an ambient temperature, a start time (MATLAB datevec), and a
data struct whose fields depend on the cycle type.

Output: one row per (battery_id, discharge cycle) with the measured
discharge capacity, within-cycle measurement summaries, and summaries of
the immediately preceding charge cycle. Impedance cycles are excluded:
they are a different measurement modality (EIS, giving Re/Rct), are
irregularly interleaved, and are absent for long stretches on several
batteries -- treating them as an optional enrichment rather than a core
column keeps the table dense. Revisit if impedance features are wanted.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = Path("data/raw/extracted/5. Battery Data Set")
DEFAULT_OUT_PATH = Path("data/processed/cycles.parquet")

# Cells are rated 2.0 Ah; a parsed capacity outside this range means the
# parse grabbed the wrong field, not that the battery is unusual.
CAPACITY_VALID_RANGE_AH = (0.0, 3.2)


class DataValidationError(Exception):
    """Parsed output violates an expected invariant."""


def matlab_datevec_to_datetime(vec: np.ndarray) -> datetime:
    """Convert a MATLAB 6-element datevec [y, m, d, H, M, S.frac] to datetime.

    Args:
        vec: Array with at least 6 elements; seconds may be fractional.

    Raises:
        ValueError: If vec has fewer than 6 elements.
    """
    flat = np.asarray(vec).ravel()
    if flat.size < 6:
        raise ValueError(f"expected 6-element datevec, got shape {vec.shape}")
    year, month, day, hour, minute = (int(x) for x in flat[:5])
    return datetime(year, month, day, hour, minute) + timedelta(seconds=float(flat[5]))


def _series(data: np.ndarray, field: str) -> np.ndarray:
    """Extract a 1-d float array field from a cycle's data struct."""
    return np.asarray(data[field][0, 0], dtype=float).ravel()


def _scalar_capacity(data: np.ndarray) -> float | None:
    """Pull the measured capacity from a discharge cycle's data struct.

    Returns None when the field is missing or empty (aborted run), so the
    caller can skip the cycle explicitly instead of writing a NaN row.
    """
    names = data.dtype.names or ()
    if "Capacity" not in names:
        return None
    cap = np.asarray(data["Capacity"][0, 0], dtype=float).ravel()
    if cap.size == 0:
        return None
    return float(cap[0])


def summarize_discharge(data: np.ndarray) -> dict[str, float]:
    """Within-cycle summary stats for one discharge cycle.

    All values are computed from that cycle's own measurements only, so
    they are safe inputs for later feature engineering.
    """
    voltage = _series(data, "Voltage_measured")
    current = _series(data, "Current_measured")
    temp = _series(data, "Temperature_measured")
    time_s = _series(data, "Time")
    return {
        "discharge_duration_s": float(time_s[-1]),
        "voltage_min": float(voltage.min()),
        "voltage_mean": float(voltage.mean()),
        "current_mean": float(current.mean()),
        "temp_max": float(temp.max()),
        "temp_mean": float(temp.mean()),
    }


def summarize_charge(data: np.ndarray) -> dict[str, float]:
    """Within-cycle summary stats for one charge cycle.

    Charge behaviour degrades with age (longer CV phase, higher taper
    current), so we keep the terminal current alongside duration and
    thermal summaries. The final measured current approximates the CV
    taper current at cutoff.
    """
    current = _series(data, "Current_measured")
    temp = _series(data, "Temperature_measured")
    time_s = _series(data, "Time")
    return {
        "charge_duration_s": float(time_s[-1]),
        "charge_current_end": float(current[-1]),
        "charge_temp_max": float(temp.max()),
        "charge_temp_mean": float(temp.mean()),
    }


def parse_battery_file(path: Path) -> pd.DataFrame:
    """Parse one battery .mat file into a per-discharge-cycle frame.

    Walks the cycle array in file order (which is chronological), keeping
    the latest seen charge-cycle summary and attaching it to the next
    discharge row. Discharges with no measured capacity are skipped with
    a warning rather than emitted as NaN rows.

    Args:
        path: Path to a B00xx.mat file; the stem is used as battery_id.

    Raises:
        KeyError: If the file does not contain the expected top-level struct.
    """
    battery_id = path.stem
    mat = sio.loadmat(str(path))
    if battery_id not in mat:
        raise KeyError(f"{path.name}: expected top-level key {battery_id!r}")
    cycles = mat[battery_id]["cycle"][0, 0]
    n_cycles = cycles.shape[1]

    rows: list[dict[str, object]] = []
    last_charge: dict[str, float] | None = None
    n_skipped = 0
    for i in range(n_cycles):
        ctype = str(cycles["type"][0, i][0])
        data = cycles["data"][0, i]
        if ctype == "charge":
            last_charge = summarize_charge(data)
            continue
        if ctype == "impedance":
            continue
        if ctype != "discharge":
            logger.warning("%s: unknown cycle type %r at index %d, skipping", battery_id, ctype, i)
            continue

        capacity = _scalar_capacity(data)
        if capacity is None:
            n_skipped += 1
            logger.warning("%s: discharge at file index %d has no capacity, skipping", battery_id, i)
            continue
        # Some runs log Capacity = 0.0: partial discharges that never reached
        # the voltage cutoff (their voltage_min sits at 2.9-3.8 V instead of
        # the ~2.5 V cutoff). Zero is not a capacity measurement, so these are
        # skipped the same way as cycles with no Capacity field at all.
        if capacity <= 0:
            n_skipped += 1
            logger.warning(
                "%s: discharge at file index %d has non-positive capacity %.3f "
                "(aborted/partial run), skipping", battery_id, i, capacity,
            )
            continue

        row: dict[str, object] = {
            "battery_id": battery_id,
            "file_cycle_number": i,
            "cycle_start_time": matlab_datevec_to_datetime(cycles["time"][0, i]),
            "ambient_temperature": float(cycles["ambient_temperature"][0, i][0, 0]),
            "capacity_ah": capacity,
        }
        row.update(summarize_discharge(data))
        # A discharge with no preceding charge in the file (can happen at the
        # start of a batch) gets NaN charge columns rather than being dropped.
        row.update(last_charge or {k: np.nan for k in
                   ("charge_duration_s", "charge_current_end", "charge_temp_max", "charge_temp_mean")})
        rows.append(row)
        last_charge = None

    df = pd.DataFrame(rows)
    df["cycle_index"] = np.arange(1, len(df) + 1)
    if n_skipped:
        logger.info("%s: skipped %d capacity-less discharge cycles", battery_id, n_skipped)
    logger.info("%s: %d discharge rows from %d raw cycle entries", battery_id, len(df), n_cycles)
    return df


def discover_battery_files(raw_root: Path) -> dict[str, Path]:
    """Find one .mat file per battery id under raw_root.

    The NASA release ships B0025-B0028 twice (batch 2 and batch 3, byte
    identical). When duplicates appear we keep the larger file (they tie
    in practice; ties resolve to the first in sorted path order) and log
    the skipped copy.
    """
    chosen: dict[str, Path] = {}
    for path in sorted(raw_root.rglob("*.mat")):
        bid = path.stem
        if bid in chosen:
            keep, skip = chosen[bid], path
            if path.stat().st_size > chosen[bid].stat().st_size:
                keep, skip = path, chosen[bid]
            chosen[bid] = keep
            logger.info("%s: duplicate file, keeping %s, skipping %s", bid, keep, skip)
        else:
            chosen[bid] = path
    if not chosen:
        raise FileNotFoundError(f"no .mat files found under {raw_root}")
    return chosen


def validate_cycles_table(df: pd.DataFrame) -> None:
    """Raise DataValidationError if the parsed table violates invariants."""
    if df.empty:
        raise DataValidationError("parsed table is empty")
    if df[["battery_id", "cycle_index", "capacity_ah"]].isna().any().any():
        raise DataValidationError("null battery_id, cycle_index or capacity_ah present")
    if df.duplicated(subset=["battery_id", "cycle_index"]).any():
        raise DataValidationError("duplicate (battery_id, cycle_index) rows")
    lo, hi = CAPACITY_VALID_RANGE_AH
    bad = df[~df["capacity_ah"].between(lo, hi, inclusive="neither")]
    if not bad.empty:
        raise DataValidationError(
            f"{len(bad)} rows with capacity outside ({lo}, {hi}) Ah, "
            f"e.g. {bad[['battery_id', 'cycle_index', 'capacity_ah']].head().to_dict('records')}"
        )
    if (df["discharge_duration_s"] <= 0).any():
        raise DataValidationError("non-positive discharge duration present")


def parse_all(raw_root: Path, out_path: Path | None = None) -> pd.DataFrame:
    """Parse every battery under raw_root into one validated table.

    Args:
        raw_root: Directory containing the extracted NASA batch folders.
        out_path: If given, the table is written there as parquet.

    Returns:
        Frame with one row per (battery_id, discharge cycle), sorted by
        battery_id then cycle_index.
    """
    files = discover_battery_files(raw_root)
    frames = [parse_battery_file(path) for _, path in sorted(files.items())]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
    validate_cycles_table(df)
    logger.info("parsed %d batteries, %d discharge rows total", df["battery_id"].nunique(), len(df))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("wrote %s", out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NASA battery .mat files to parquet")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parse_all(args.raw_root, args.out)


if __name__ == "__main__":
    main()
