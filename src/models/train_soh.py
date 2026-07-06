"""SoH regression at multiple horizons: baselines first, models second.

Task formulation. A row's features encode information available at cycle
t: past-SoH lags (up to t-1) plus pre-discharge telemetry of cycle t
(charge phase, ambient, cycle position, rest time). The horizon-k target
is the SoH measured k discharge cycles after the last observed SoH, i.e.
soh(t + k - 1); k=1 is nowcasting the current cycle.

Why multiple horizons: at k=1 the last-value and global-slope baselines
are near-optimal (SoH moves ~0.003/cycle) and no model beats them -- that
result stays in the report as the honesty floor. The model's genuine job
is k >= 10, where a flat last-value prediction keeps paying the full fade
and a global slope ignores battery-specific degradation dynamics. Each
horizon gets its own directly-trained model (direct strategy) rather than
recursive rollout, so horizon errors are independent and comparable.

Evaluation protocol:
- Rows where the last-value baseline is undefined (first cycle per
  battery) or the horizon target does not exist (end of life reached)
  are excluded for EVERY method, so all methods score identical rows.
- Metrics are per held-out battery, never aggregate-only.
- R squared above R2_LEAKAGE_TRIPWIRE on unseen batteries is logged as a
  leakage suspect to audit, not celebrated. Audit outcome for k=1: the
  baselines themselves exceed it, so saturation comes from target
  autocorrelation, not leakage (features are prefix-invariance tested).
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splits import split_frames
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_ARTIFACT_ROOT = Path("artifacts")

PAST_SOH_FEATURES: tuple[str, ...] = ("soh_prev", "soh_roll_mean", "soh_roll_std", "fade_rate")
TELEMETRY_FEATURES: list[str] = [c for c in FEATURE_COLUMNS if c not in PAST_SOH_FEATURES]

HORIZONS: tuple[int, ...] = (1, 10, 25, 50)

# Anything above this on unseen batteries is treated as a leakage suspect
# to audit, per the project ground rules -- not as a result.
R2_LEAKAGE_TRIPWIRE = 0.95


@dataclass(frozen=True)
class TrainConfig:
    version: str = "0.2.0"
    seed: int = 42
    horizons: tuple[int, ...] = HORIZONS
    lgbm_params: dict[str, object] = field(
        default_factory=lambda: {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "verbosity": -1,
        }
    )
    early_stopping_rounds: int = 50


def horizon_target(k: int) -> str:
    return f"soh_h{k}"


def add_horizon_targets(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Attach soh(t + k - 1) per battery for each horizon k.

    shift(-(k-1)) pulls FUTURE values into the target column, which is the
    one and only place future information is allowed: it is the label.
    Rows near end of life, where the horizon extends past the data, get
    NaN and are excluded from evaluation for that horizon.
    """
    df = df.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True).copy()
    grouped = df.groupby("battery_id", sort=False)
    for k in horizons:
        df[horizon_target(k)] = grouped[TARGET_COLUMN].shift(-(k - 1))
    return df


def evaluable(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Rows where both the last-value baseline and the horizon-k label exist."""
    return df[df["soh_prev"].notna() & df[horizon_target(k)].notna()]


def per_battery_metrics(
    df: pd.DataFrame, y_pred: np.ndarray, method: str, k: int
) -> pd.DataFrame:
    """MAE / RMSE / R2 per battery for one prediction vector at horizon k."""
    out = df[["battery_id", horizon_target(k)]].copy()
    out["pred"] = y_pred
    rows = []
    for bid, g in out.groupby("battery_id"):
        rows.append(
            {
                "method": method,
                "horizon": k,
                "battery_id": bid,
                "n": len(g),
                "mae": mean_absolute_error(g[horizon_target(k)], g["pred"]),
                "rmse": float(np.sqrt(mean_squared_error(g[horizon_target(k)], g["pred"]))),
                "r2": r2_score(g[horizon_target(k)], g["pred"]),
            }
        )
    return pd.DataFrame(rows)


def baseline_last_value(df: pd.DataFrame) -> np.ndarray:
    """Predict the last observed SoH, flat at every horizon. The floor."""
    return df["soh_prev"].to_numpy()


def global_fade_slope(train: pd.DataFrame) -> float:
    """Mean per-cycle SoH change across training batteries; one global number."""
    return float(
        train.sort_values(["battery_id", "cycle_index"])
        .groupby("battery_id")[TARGET_COLUMN]
        .diff()
        .mean()
    )


def baseline_global_slope(df: pd.DataFrame, slope: float, k: int) -> np.ndarray:
    """Last observed SoH plus k steps of the train-estimated global fade."""
    return df["soh_prev"].to_numpy() + slope * k


def train_ridge(
    train: pd.DataFrame, features: list[str], target: str, seed: int
) -> Pipeline:
    """Median-impute (train statistics only), scale, ridge-regress."""
    fit_rows = train[train[target].notna()]
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=seed)),
        ]
    )
    model.fit(fit_rows[features], fit_rows[target])
    return model


def train_lgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    target: str,
    config: TrainConfig,
) -> lgb.LGBMRegressor:
    """Gradient boosting with early stopping on the held-out val battery."""
    fit_rows = train[train[target].notna()]
    val_rows = val[val[target].notna()]
    model = lgb.LGBMRegressor(random_state=config.seed, **config.lgbm_params)
    model.fit(
        fit_rows[features],
        fit_rows[target],
        eval_set=[(val_rows[features], val_rows[target])],
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
    return model


def save_artifacts(
    models: dict[str, object],
    config: TrainConfig,
    metrics: pd.DataFrame,
    artifact_root: Path,
) -> Path:
    """Persist models, config and metrics as a versioned artifact directory.

    exist_ok=False: overwriting a version in place would destroy the audit
    trail; bump the version instead.
    """
    artifact_dir = artifact_root / f"soh_v{config.version}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        joblib.dump(model, artifact_dir / f"{name}.joblib")
    (artifact_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))
    metrics.to_json(artifact_dir / "metrics.json", orient="records", indent=2)
    logger.info("artifacts written to %s", artifact_dir)
    return artifact_dir


def run(
    features_path: Path = DEFAULT_FEATURES_PATH,
    artifact_root: Path | None = DEFAULT_ARTIFACT_ROOT,
    config: TrainConfig = TrainConfig(),
) -> pd.DataFrame:
    """Train baselines and per-horizon models, return per-battery metrics."""
    df = add_horizon_targets(pd.read_parquet(features_path), config.horizons)
    frames = split_frames(df)
    train, val, test = frames["train"], frames["val"], frames["test"]
    slope = global_fade_slope(train)
    logger.info("global fade slope (train): %.6f SoH/cycle", slope)

    variants: dict[str, list[str]] = {
        "ridge_autoregressive": FEATURE_COLUMNS,
        "lgbm_autoregressive": FEATURE_COLUMNS,
        "lgbm_telemetry": TELEMETRY_FEATURES,
    }

    models: dict[str, object] = {}
    all_metrics: list[pd.DataFrame] = []
    for k in config.horizons:
        target = horizon_target(k)
        for name, features in variants.items():
            if name.startswith("ridge"):
                model: object = train_ridge(train, features, target, config.seed)
            else:
                model = train_lgbm(train, val, features, target, config)
            models[f"{name}_h{k}"] = model

        for split_name, frame in (("val", val), ("test", test)):
            ev = evaluable(frame, k)
            preds: dict[str, np.ndarray] = {
                "baseline_last_value": baseline_last_value(ev),
                "baseline_global_slope": baseline_global_slope(ev, slope, k),
            }
            for name, features in variants.items():
                model = models[f"{name}_h{k}"]
                preds[name] = np.asarray(model.predict(ev[features]))  # type: ignore[attr-defined]
            for method, y_pred in preds.items():
                m = per_battery_metrics(ev, y_pred, method, k)
                m.insert(0, "split", split_name)
                all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    suspects = metrics[
        (metrics["r2"] > R2_LEAKAGE_TRIPWIRE) & (~metrics["method"].str.startswith("baseline"))
    ]
    if not suspects.empty:
        logger.warning(
            "R2 above %.2f on unseen batteries for: %s -- treat as leakage suspect and audit "
            "before reporting (decisive check is MAE vs baselines, see module docstring)",
            R2_LEAKAGE_TRIPWIRE,
            suspects[["split", "method", "horizon", "battery_id", "r2"]].to_dict("records"),
        )
    if artifact_root is not None:
        save_artifacts(models, config, metrics, artifact_root)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SoH baselines and models")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    metrics = run(args.features, args.artifacts)
    pd.set_option("display.width", 200)
    print(metrics.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
