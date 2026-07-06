"""SoH regression: naive baselines first, then linear and gradient boosting.

Evaluation protocol:
- Rows with no previous observed SoH (each battery's first cycle) are
  excluded from EVALUATION for every method, because the last-value
  baseline is undefined there and models must be compared on identical
  rows. Models still TRAIN on all rows.
- Metrics are reported per held-out battery, never aggregate-only.
- R squared is reported as a tripwire, not a headline: with soh_prev as a
  feature (or as the baseline itself), one-step-ahead SoH tracks the
  previous value closely and R squared saturates by construction. The
  honest comparison is MAE/RMSE against the last-value baseline.

Two feature sets are trained deliberately:
- autoregressive: all leakage-safe features including past-SoH lags.
- telemetry-only: past-SoH lags removed; shows what charge-phase and
  operating-condition telemetry alone is worth, which is the honest
  measure of the sensor signal (and the harder claim).
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

# Anything above this on unseen batteries is treated as a leakage suspect
# to audit, per the project ground rules -- not as a result.
R2_LEAKAGE_TRIPWIRE = 0.95


@dataclass(frozen=True)
class TrainConfig:
    version: str = "0.1.0"
    seed: int = 42
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


def evaluable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows on which every method (incl. the last-value baseline) is defined."""
    return df[df["soh_prev"].notna()]


def per_battery_metrics(df: pd.DataFrame, y_pred: np.ndarray, method: str) -> pd.DataFrame:
    """MAE / RMSE / R2 per battery for one prediction vector."""
    out = df[["battery_id", TARGET_COLUMN]].copy()
    out["pred"] = y_pred
    rows = []
    for bid, g in out.groupby("battery_id"):
        rows.append(
            {
                "method": method,
                "battery_id": bid,
                "n": len(g),
                "mae": mean_absolute_error(g[TARGET_COLUMN], g["pred"]),
                "rmse": float(np.sqrt(mean_squared_error(g[TARGET_COLUMN], g["pred"]))),
                "r2": r2_score(g[TARGET_COLUMN], g["pred"]),
            }
        )
    return pd.DataFrame(rows)


def baseline_last_value(df: pd.DataFrame) -> np.ndarray:
    """Predict the last observed SoH. THE floor every model must beat."""
    return df["soh_prev"].to_numpy()


def baseline_global_slope(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """Last observed SoH plus the mean per-cycle fade estimated on train.

    The slope is the mean of per-cycle SoH differences across training
    batteries -- one global number, no per-battery fitting.
    """
    slope = (
        train.sort_values(["battery_id", "cycle_index"])
        .groupby("battery_id")[TARGET_COLUMN]
        .diff()
        .mean()
    )
    return df["soh_prev"].to_numpy() + slope


def train_ridge(train: pd.DataFrame, features: list[str], seed: int) -> Pipeline:
    """Median-impute (train statistics only), scale, ridge-regress."""
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=seed)),
        ]
    )
    model.fit(train[features], train[TARGET_COLUMN])
    return model


def train_lgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    config: TrainConfig,
) -> lgb.LGBMRegressor:
    """Gradient boosting with early stopping on the held-out val battery."""
    model = lgb.LGBMRegressor(random_state=config.seed, **config.lgbm_params)
    model.fit(
        train[features],
        train[TARGET_COLUMN],
        eval_set=[(val[features], val[TARGET_COLUMN])],
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
    """Train baselines and models, return per-battery metrics on val and test."""
    df = pd.read_parquet(features_path)
    frames = split_frames(df)
    train, val, test = frames["train"], frames["val"], frames["test"]

    models: dict[str, object] = {
        "ridge_autoregressive": train_ridge(train, FEATURE_COLUMNS, config.seed),
        "ridge_telemetry": train_ridge(train, TELEMETRY_FEATURES, config.seed),
        "lgbm_autoregressive": train_lgbm(train, val, FEATURE_COLUMNS, config),
        "lgbm_telemetry": train_lgbm(train, val, TELEMETRY_FEATURES, config),
    }
    feature_set = {
        "ridge_autoregressive": FEATURE_COLUMNS,
        "ridge_telemetry": TELEMETRY_FEATURES,
        "lgbm_autoregressive": FEATURE_COLUMNS,
        "lgbm_telemetry": TELEMETRY_FEATURES,
    }

    all_metrics: list[pd.DataFrame] = []
    for split_name, frame in (("val", val), ("test", test)):
        ev = evaluable(frame)
        preds: dict[str, np.ndarray] = {
            "baseline_last_value": baseline_last_value(ev),
            "baseline_global_slope": baseline_global_slope(ev, train),
        }
        for name, model in models.items():
            preds[name] = np.asarray(model.predict(ev[feature_set[name]]))  # type: ignore[attr-defined]
        for method, y_pred in preds.items():
            m = per_battery_metrics(ev, y_pred, method)
            m.insert(0, "split", split_name)
            all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    suspects = metrics[(metrics["r2"] > R2_LEAKAGE_TRIPWIRE) & (~metrics["method"].str.startswith("baseline"))]
    if not suspects.empty:
        logger.warning(
            "R2 above %.2f on unseen batteries for: %s -- treat as leakage suspect and audit "
            "before reporting (see module docstring: R2 saturates by construction when past "
            "SoH is available; the decisive check is MAE vs baseline_last_value)",
            R2_LEAKAGE_TRIPWIRE,
            suspects[["split", "method", "battery_id", "r2"]].to_dict("records"),
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
