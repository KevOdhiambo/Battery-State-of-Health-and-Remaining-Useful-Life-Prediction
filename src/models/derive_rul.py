"""Derive RUL from projected SoH trajectories, reported separately from SoH.

RUL is NOT modeled as its own regressor. Standing at a prediction point
(a feature row, whose information set is past SoH through cycle t-1 plus
the pre-discharge telemetry of cycle t), we project the SoH trajectory
forward and read off the first crossing of the EOL threshold. Predicted
RUL is expressed in cycles after the last observed SoH measurement, so it
is directly comparable to true RUL = first cycle with SoH below the
threshold minus the last observed cycle.

Thresholds. Primary EOL is 70 percent of rated capacity (1.4 Ah): it is
the fade level NASA ran the canonical batch to, so every non-censored
battery's true EOL is observed rather than extrapolated. 80 percent is
reported as sensitivity -- and because the validation battery B0007 never
reaches 70 percent in its recorded life (min SoH 0.7002), it is right-
censored at the primary threshold and only evaluable at 80 percent. That
censoring is reported, not hidden.

Two projection methods, mirroring the Stage 5 result that neither
dominates on both held-out batteries:
- global_slope: last observed SoH plus the train-estimated mean fade per
  cycle, solved for the threshold crossing in closed form.
- lgbm: direct per-horizon models on a dense horizon grid; the predicted
  trajectory is piecewise linear between grid points, first crossing
  interpolated. A trajectory that never crosses within the grid is
  extrapolated with its last segment's slope; if that slope is
  non-negative the prediction is censored and counted, not scored.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import split_frames
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.train_soh import (
    TrainConfig,
    add_horizon_targets,
    global_fade_slope,
    horizon_target,
    train_lgbm,
)

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_OUT_PATH = Path("artifacts/rul_v0.1.0/rul_metrics.json")

EOL_THRESHOLDS: tuple[float, ...] = (0.70, 0.80)
# Dense grid for trajectory projection. Step 5 keeps 25 model fits while
# bounding interpolation error at +/- 2.5 cycles, well under RUL noise.
HORIZON_GRID: tuple[int, ...] = tuple(range(1, 122, 5))
# Skip the first cycles of each battery: rolling features are NaN-heavy
# and no operator asks for an EOL forecast on a fresh cell's first cycles.
MIN_HISTORY_CYCLES = 10


@dataclass(frozen=True)
class RulPrediction:
    battery_id: str
    cycle_index: int
    method: str
    threshold: float
    rul_true: float
    rul_pred: float | None  # None = trajectory never crosses (censored)


def true_eol_cycle(battery: pd.DataFrame, threshold: float) -> int | None:
    """First cycle_index whose observed SoH is below threshold, else None."""
    below = battery.loc[battery[TARGET_COLUMN] < threshold, "cycle_index"]
    return int(below.min()) if not below.empty else None


def crossing_from_trajectory(
    horizons: np.ndarray, soh_pred: np.ndarray, threshold: float
) -> float | None:
    """First threshold crossing of a piecewise-linear predicted trajectory.

    Returns the (fractional) horizon of the crossing, extrapolating past
    the grid with the last segment's slope when needed, or None when the
    trajectory never trends below the threshold.
    """
    below = np.nonzero(soh_pred < threshold)[0]
    if below.size:
        j = int(below[0])
        if j == 0:
            return float(horizons[0])
        x0, x1 = horizons[j - 1], horizons[j]
        y0, y1 = soh_pred[j - 1], soh_pred[j]
        return float(x0 + (y0 - threshold) * (x1 - x0) / (y0 - y1))
    tail_slope = (soh_pred[-1] - soh_pred[-2]) / (horizons[-1] - horizons[-2])
    if tail_slope >= 0:
        return None
    return float(horizons[-1] + (soh_pred[-1] - threshold) / -tail_slope)


def slope_rul(last_soh: float, slope: float, threshold: float) -> float | None:
    """Closed-form crossing for the constant-fade baseline."""
    if slope >= 0:
        return None
    if last_soh < threshold:
        return 0.0
    return (threshold - last_soh) / slope


def fit_horizon_models(
    train: pd.DataFrame, val: pd.DataFrame, config: TrainConfig
) -> dict[int, object]:
    """One direct LGBM per grid horizon, early-stopped on the val battery."""
    models: dict[int, object] = {}
    for k in HORIZON_GRID:
        target = horizon_target(k)
        if train[target].notna().sum() < 50:
            logger.info("horizon %d: under 50 training rows, stopping grid here", k)
            break
        models[k] = train_lgbm(train, val, FEATURE_COLUMNS, target, config)
    logger.info("fitted %d horizon models (k=%d..%d)", len(models), min(models), max(models))
    return models


def predict_rul(
    battery: pd.DataFrame,
    models: dict[int, object],
    slope: float,
    thresholds: tuple[float, ...] = EOL_THRESHOLDS,
) -> list[RulPrediction]:
    """RUL predictions at every valid standing point of one battery."""
    horizons = np.array(sorted(models), dtype=float)
    preds: list[RulPrediction] = []
    eol: dict[float, int | None] = {t: true_eol_cycle(battery, t) for t in thresholds}

    rows = battery[
        (battery["cycle_index"] >= MIN_HISTORY_CYCLES) & battery["soh_prev"].notna()
    ]
    if rows.empty:
        return preds
    trajectory = np.column_stack(
        [np.asarray(models[int(k)].predict(rows[FEATURE_COLUMNS])) for k in horizons]  # type: ignore[attr-defined]
    )
    for (_, row), soh_traj in zip(rows.iterrows(), trajectory, strict=True):
        last_observed = int(row["cycle_index"]) - 1
        for threshold in thresholds:
            if eol[threshold] is None or last_observed >= eol[threshold]:
                continue
            rul_true = float(eol[threshold] - last_observed)
            preds.append(
                RulPrediction(
                    battery_id=str(row["battery_id"]),
                    cycle_index=int(row["cycle_index"]),
                    method="lgbm_trajectory",
                    threshold=threshold,
                    rul_true=rul_true,
                    rul_pred=crossing_from_trajectory(horizons, soh_traj, threshold),
                )
            )
            preds.append(
                RulPrediction(
                    battery_id=str(row["battery_id"]),
                    cycle_index=int(row["cycle_index"]),
                    method="global_slope",
                    threshold=threshold,
                    rul_true=rul_true,
                    rul_pred=slope_rul(float(row["soh_prev"]), slope, threshold),
                )
            )
    return preds


def summarize(preds: list[RulPrediction]) -> pd.DataFrame:
    """Per (battery, threshold, method): MAE in cycles and censoring counts."""
    df = pd.DataFrame([p.__dict__ for p in preds])
    if df.empty:
        return df
    rows = []
    for (bid, thr, method), g in df.groupby(["battery_id", "threshold", "method"]):
        scored = g[g["rul_pred"].notna()]
        err = (scored["rul_pred"] - scored["rul_true"]).abs()
        rows.append(
            {
                "battery_id": bid,
                "threshold": thr,
                "method": method,
                "n_points": len(g),
                "n_censored_preds": int(g["rul_pred"].isna().sum()),
                "rul_mae_cycles": float(err.mean()),
                "rul_median_ae_cycles": float(err.median()),
                "rul_max_ae_cycles": float(err.max()),
            }
        )
    return pd.DataFrame(rows)


def run(
    features_path: Path = DEFAULT_FEATURES_PATH,
    out_path: Path | None = DEFAULT_OUT_PATH,
    config: TrainConfig = TrainConfig(version="0.2.0", horizons=HORIZON_GRID),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit grid models on train, derive RUL on val and test batteries.

    Returns:
        (per-point predictions, per-battery summary).
    """
    df = add_horizon_targets(pd.read_parquet(features_path), HORIZON_GRID)
    frames = split_frames(df)
    train, val, test = frames["train"], frames["val"], frames["test"]
    slope = global_fade_slope(train)
    models = fit_horizon_models(train, val, config)

    preds: list[RulPrediction] = []
    for frame in (val, test):
        for _, battery in frame.groupby("battery_id"):
            preds.extend(predict_rul(battery.sort_values("cycle_index"), models, slope))
    for threshold in EOL_THRESHOLDS:
        for frame, name in ((val, "val"), (test, "test")):
            for bid, g in frame.groupby("battery_id"):
                if true_eol_cycle(g, threshold) is None:
                    logger.warning(
                        "%s (%s) never crosses SoH %.2f in its observed life -- "
                        "right-censored, no RUL evaluation at this threshold",
                        bid, name, threshold,
                    )

    pred_df = pd.DataFrame([p.__dict__ for p in preds])
    summary = summarize(preds)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "summary": summary.to_dict("records"),
                    "predictions": pred_df.to_dict("records"),
                },
                indent=2,
            )
        )
        logger.info("wrote %s", out_path)
    return pred_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive RUL from projected SoH trajectories")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _, summary = run(args.features, args.out)
    pd.set_option("display.width", 200)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
