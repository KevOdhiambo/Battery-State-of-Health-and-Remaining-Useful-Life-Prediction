"""Streamlit demo: SoH projection, derived RUL, and drift for one battery.

Disposable presentation layer. It imports the SAME pipeline code the
training run uses (features, splits, trajectory projection, drift) so
there is no reimplemented inference path to drift out of sync -- but
nothing in src/ depends on this file, and the repo stands alone if the
demo is deleted.

Run from the repo root:
    streamlit run demo/app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.splits import split_frames
from src.eval.report import (
    COLOR_ACTUAL,
    COLOR_BASELINE,
    COLOR_INK,
    COLOR_INK_MUTED,
    COLOR_MODEL,
    _style_axes,
)
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.derive_rul import (
    HORIZON_GRID,
    crossing_from_trajectory,
    fit_horizon_models,
    slope_rul,
    true_eol_cycle,
)
from src.models.train_soh import TrainConfig, add_horizon_targets, global_fade_slope
from src.monitoring.drift import drift_report

FEATURES_PATH = Path("data/processed/features.parquet")
MIN_STANDING_CYCLE = 10


@st.cache_resource(show_spinner="Fitting horizon models on training batteries...")
def load_pipeline() -> tuple[dict[str, pd.DataFrame], dict[int, object], float]:
    df = add_horizon_targets(pd.read_parquet(FEATURES_PATH), HORIZON_GRID)
    frames = split_frames(df)
    slope = global_fade_slope(frames["train"])
    models = fit_horizon_models(frames["train"], frames["val"], TrainConfig(horizons=HORIZON_GRID))
    return frames, models, slope


def projection_figure(
    battery: pd.DataFrame,
    row: pd.DataFrame,
    horizons: np.ndarray,
    traj: np.ndarray,
    baseline: np.ndarray,
    standing_cycle: int,
    threshold: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=120)
    observed = battery[battery["cycle_index"] <= standing_cycle]
    future = battery[battery["cycle_index"] >= standing_cycle]
    ax.plot(observed["cycle_index"], observed[TARGET_COLUMN], color=COLOR_ACTUAL,
            linewidth=2, label="Observed SoH")
    ax.plot(future["cycle_index"], future[TARGET_COLUMN], color=COLOR_ACTUAL,
            linewidth=1.2, alpha=0.35, label="Actual future (hidden from model)")
    x = standing_cycle - 1 + horizons
    ax.plot(x, traj, color=COLOR_MODEL, linewidth=2, label="Model projection")
    ax.plot(x, baseline, color=COLOR_BASELINE, linewidth=1.4, linestyle=(0, (4, 3)),
            label="Slope baseline")
    ax.axhline(threshold, color="#9AA3AF", linewidth=1, linestyle=(0, (4, 4)))
    ax.text(battery["cycle_index"].min() + 1, threshold + 0.004,
            f"EOL threshold ({threshold:.0%} of rated)", fontsize=9,
            color=COLOR_INK_MUTED, va="bottom")
    ax.plot(standing_cycle, float(row[TARGET_COLUMN].iloc[0]), marker="o", markersize=9,
            color=COLOR_ACTUAL, markerfacecolor="white", markeredgewidth=2, zorder=5)
    ax.set_xlabel("Discharge cycle number", fontsize=10, color=COLOR_INK)
    ax.set_ylabel("SoH (capacity / 2.0 Ah rated)", fontsize=10, color=COLOR_INK)
    legend = ax.legend(loc="lower left", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(COLOR_INK)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Battery SoH / RUL", layout="wide")
    st.title("Battery SoH and RUL -- held-out battery explorer")
    st.caption(
        "Models trained on other batteries only (unit-level split). "
        "Pick a standing point: the model sees history up to that cycle and "
        "projects the rest of the SoH trajectory."
    )

    frames, models, slope = load_pipeline()
    horizons = np.array(sorted(models), dtype=float)
    eval_batteries = {
        "B0007 (validation battery)": frames["val"],
        "B0018 (test battery)": frames["test"],
    }

    with st.sidebar:
        choice = st.selectbox("Held-out battery", list(eval_batteries))
        battery = eval_batteries[choice].sort_values("cycle_index")
        max_cycle = int(battery["cycle_index"].max())
        standing_cycle = st.slider(
            "Standing point (last cycle the model can see)",
            MIN_STANDING_CYCLE, max_cycle - 1, value=int(max_cycle * 0.4),
        )
        threshold = st.radio(
            "EOL threshold", (0.70, 0.80), format_func=lambda t: f"{t:.0%} of rated capacity"
        )

    row = battery[battery["cycle_index"] == standing_cycle]
    if row.empty or row["soh_prev"].isna().all():
        st.warning("No usable feature row at this cycle; pick another standing point.")
        return

    traj = np.array(
        [float(models[int(k)].predict(row[FEATURE_COLUMNS])[0]) for k in horizons]  # type: ignore[attr-defined]
    )
    baseline = float(row["soh_prev"].iloc[0]) + slope * horizons

    rul_model = crossing_from_trajectory(horizons, traj, float(threshold))
    rul_base = slope_rul(float(row["soh_prev"].iloc[0]), slope, float(threshold))
    eol = true_eol_cycle(battery, float(threshold))
    rul_true = float(eol - (standing_cycle - 1)) if eol is not None and eol > standing_cycle - 1 else None

    st.pyplot(
        projection_figure(battery, row, horizons, traj, baseline, standing_cycle, float(threshold))
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("True RUL (cycles)", "censored" if rul_true is None else f"{rul_true:.0f}")
    c2.metric(
        "Model RUL",
        "no crossing" if rul_model is None else f"{rul_model:.0f}",
        None if rul_model is None or rul_true is None else f"{rul_model - rul_true:+.0f} vs true",
        delta_color="off",
    )
    c3.metric(
        "Baseline RUL",
        "no crossing" if rul_base is None else f"{rul_base:.0f}",
        None if rul_base is None or rul_true is None else f"{rul_base - rul_true:+.0f} vs true",
        delta_color="off",
    )
    if rul_true is None:
        st.info(
            "This battery never crosses the selected threshold in its recorded "
            "life (right-censored), so there is no true RUL to compare against."
        )

    st.subheader("Input drift vs training distribution")
    st.caption(
        "PSI on the features observed up to the standing point. A held-out "
        "battery is a different physical cell, so shift is expected; the point "
        "is the monitoring hook, not an alarm."
    )
    seen = battery[battery["cycle_index"] <= standing_cycle]
    drift = drift_report(frames["train"], seen, FEATURE_COLUMNS)
    worst = drift.iloc[0]
    st.metric("Worst feature PSI", f"{worst['psi']:.2f}", worst["feature"], delta_color="off")
    st.dataframe(drift.round(3), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
