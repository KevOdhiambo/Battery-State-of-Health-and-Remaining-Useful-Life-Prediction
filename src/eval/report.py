"""Evaluation report: per-battery curves, baseline-vs-model tables, drift.

Produces reports/results.md plus one figure per held-out battery showing
the actual SoH curve against model trajectories projected from three
standing points (25 / 50 / 75 percent of observed life), with the
constant-slope baseline projection drawn from the same points. Tables are
read from the versioned training artifacts, so the report reflects what
was actually trained, not a re-run.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.splits import split_frames
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.derive_rul import HORIZON_GRID, fit_horizon_models
from src.models.train_soh import TrainConfig, add_horizon_targets, global_fade_slope
from src.monitoring.drift import drift_report

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_SOH_METRICS = Path("artifacts/soh_v0.2.0/metrics.json")
DEFAULT_RUL_METRICS = Path("artifacts/rul_v0.1.0/rul_metrics.json")
DEFAULT_REPORT_DIR = Path("reports")

STANDING_POINT_FRACTIONS = (0.25, 0.50, 0.75)
EOL_PRIMARY = 0.70

# Palette validated with the dataviz six-checks script (light surface):
# blue/orange pass all checks; gray is reserved for the de-emphasized
# dashed baseline reference, which carries a direct label and line-style
# as secondary encoding.
COLOR_ACTUAL = "#3B6EC5"
COLOR_MODEL = "#C2571B"
COLOR_BASELINE = "#6B7280"
COLOR_INK = "#1F2733"
COLOR_INK_MUTED = "#5B6472"
COLOR_GRID = "#E5E8EC"


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9CFD6")
    ax.tick_params(colors=COLOR_INK_MUTED, labelsize=9)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a frame as a GitHub markdown table without extra deps."""
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def plot_projections(
    battery: pd.DataFrame,
    models: dict[int, object],
    slope: float,
    out_path: Path,
) -> None:
    """Actual SoH curve with model and baseline projections from standing points."""
    battery = battery.sort_values("cycle_index")
    bid = battery["battery_id"].iloc[0]
    horizons = np.array(sorted(models), dtype=float)
    max_cycle = int(battery["cycle_index"].max())

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=130)
    ax.plot(
        battery["cycle_index"], battery[TARGET_COLUMN],
        color=COLOR_ACTUAL, linewidth=2, label="Actual SoH",
    )
    labeled_model = labeled_baseline = False
    for frac in STANDING_POINT_FRACTIONS:
        sp = int(round(max_cycle * frac))
        row = battery[battery["cycle_index"] == sp]
        if row.empty or row["soh_prev"].isna().all():
            continue
        traj = np.array(
            [float(models[int(k)].predict(row[FEATURE_COLUMNS])[0]) for k in horizons]  # type: ignore[attr-defined]
        )
        x = sp - 1 + horizons
        keep = x <= max_cycle + 25
        ax.plot(
            x[keep], traj[keep], color=COLOR_MODEL, linewidth=2,
            label="Model projection" if not labeled_model else None,
        )
        labeled_model = True
        last_soh = float(row["soh_prev"].iloc[0])
        base = last_soh + slope * horizons
        ax.plot(
            x[keep], base[keep], color=COLOR_BASELINE, linewidth=1.4,
            linestyle=(0, (4, 3)),
            label="Slope baseline" if not labeled_baseline else None,
        )
        labeled_baseline = True
        ax.plot(sp, float(row[TARGET_COLUMN].iloc[0]), marker="o", markersize=8,
                color=COLOR_ACTUAL, markerfacecolor="white", markeredgewidth=2, zorder=5)

    ax.axhline(EOL_PRIMARY, color="#9AA3AF", linewidth=1, linestyle=(0, (4, 4)))
    ax.text(2, EOL_PRIMARY + 0.004, "EOL threshold (70% of rated)", fontsize=9,
            color=COLOR_INK_MUTED, va="bottom")
    ax.set_xlabel("Discharge cycle number", fontsize=10, color=COLOR_INK)
    ax.set_ylabel("SoH (capacity / 2.0 Ah rated)", fontsize=10, color=COLOR_INK)
    ax.set_title(
        f"{bid}: actual SoH vs projections from 25/50/75 percent of life",
        fontsize=11, color=COLOR_INK, loc="left",
    )
    legend = ax.legend(loc="lower left", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(COLOR_INK)
    _style_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def build_report(
    features_path: Path = DEFAULT_FEATURES_PATH,
    soh_metrics_path: Path = DEFAULT_SOH_METRICS,
    rul_metrics_path: Path = DEFAULT_RUL_METRICS,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> Path:
    """Generate figures and results.md; returns the report path."""
    df = add_horizon_targets(pd.read_parquet(features_path), HORIZON_GRID)
    frames = split_frames(df)
    train, val, test = frames["train"], frames["val"], frames["test"]
    slope = global_fade_slope(train)
    models = fit_horizon_models(train, val, TrainConfig(horizons=HORIZON_GRID))

    for frame in (val, test):
        for bid, battery in frame.groupby("battery_id"):
            plot_projections(
                battery, models, slope, report_dir / "figures" / f"soh_projections_{bid}.png"
            )

    soh = pd.read_json(soh_metrics_path)
    soh_view = soh[
        ["split", "battery_id", "horizon", "method", "mae", "rmse"]
    ].sort_values(["split", "horizon", "mae"], ascending=[False, True, True])

    payload = json.loads(Path(rul_metrics_path).read_text())
    rul_summary = pd.DataFrame(payload["summary"])

    drift_parts = []
    for frame, name in ((val, "val B0007"), (test, "test B0018")):
        d = drift_report(train, frame, FEATURE_COLUMNS)
        d.insert(0, "compared_to_train", name)
        drift_parts.append(d)
    drift = pd.concat(drift_parts, ignore_index=True)

    lines = [
        "# Results",
        "",
        "All numbers are on whole held-out batteries (unit-level split; "
        "no battery appears in more than one split -- asserted in code).",
        "",
        "## SoH prediction error by horizon (MAE / RMSE in SoH units)",
        "",
        "Baselines listed first. The model only claims value where it beats them.",
        "",
        md_table(soh_view, floatfmt="{:.5f}"),
        "",
        "## RUL error (cycles)",
        "",
        "Derived by projecting the SoH trajectory to the EOL crossing; "
        "reported separately from SoH because it is the harder, higher-variance task. "
        "B0007 never crosses the 70 percent threshold in its observed life "
        "(right-censored), so it is only evaluable at the 80 percent sensitivity threshold.",
        "",
        md_table(rul_summary, floatfmt="{:.2f}"),
        "",
        "## Input drift (PSI, train vs held-out battery)",
        "",
        "PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant. Held-out batteries "
        "SHOULD show shift on age-linked features -- they are different physical cells; "
        "this table demonstrates the monitoring hook, not a deployment alarm.",
        "",
        md_table(drift, floatfmt="{:.3f}"),
        "",
        "## Figures",
        "",
        "![B0007 projections](figures/soh_projections_B0007.png)",
        "",
        "![B0018 projections](figures/soh_projections_B0018.png)",
        "",
    ]
    report_path = report_dir / "results.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_report(args.features, report_dir=args.report_dir)


if __name__ == "__main__":
    main()
