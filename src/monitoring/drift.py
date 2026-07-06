"""PSI-style input drift check on model features.

Same approach as the credit-risk validation harness: population stability
index per feature, with bins fixed by the training distribution's
quantiles. PSI < 0.1 means no significant shift, 0.1-0.25 moderate shift
worth investigating, > 0.25 significant shift where retraining is likely
needed. The check is deliberately model-free: it flags input shift before
ground-truth SoH is available to measure performance decay.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BINS = 10
# Laplace-style floor so an empty bin contributes a large-but-finite term
# instead of an infinite one.
PROPORTION_FLOOR = 1e-4
# Relative half-width of the tolerance band used when a training feature
# is constant and quantile bins collapse.
CONSTANT_BAND_REL = 1e-6

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = DEFAULT_BINS
) -> float:
    """PSI between a reference (training) sample and a current sample.

    Bin edges are the reference sample's quantiles, so each reference bin
    holds ~1/bins of the training mass; outer edges are open-ended to
    catch values outside the training range.

    Args:
        expected: Reference distribution (training data), NaNs dropped.
        actual: Current distribution to compare, NaNs dropped.
        bins: Number of quantile bins.

    Raises:
        ValueError: If either sample has no non-null values.
    """
    exp = expected.dropna().to_numpy(dtype=float)
    act = actual.dropna().to_numpy(dtype=float)
    if exp.size == 0 or act.size == 0:
        raise ValueError("PSI requires non-empty samples on both sides")

    edges = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if edges.size >= 3:
        inner = edges[1:-1]
    else:
        # Near-constant feature in training: three bins -- below, inside,
        # above a tight band around the constant -- so any movement lands
        # outside the band and registers as a shift.
        center = float(edges[0])
        band = max(abs(center), 1.0) * CONSTANT_BAND_REL
        inner = np.array([center - band, center + band])

    exp_counts = np.histogram(exp, bins=np.concatenate(([-np.inf], inner, [np.inf])))[0]
    act_counts = np.histogram(act, bins=np.concatenate(([-np.inf], inner, [np.inf])))[0]
    exp_prop = np.maximum(exp_counts / exp.size, PROPORTION_FLOOR)
    act_prop = np.maximum(act_counts / act.size, PROPORTION_FLOOR)
    return float(np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop)))


def rate_psi(value: float) -> str:
    if value < PSI_MODERATE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate shift"
    return "significant shift"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    bins: int = DEFAULT_BINS,
) -> pd.DataFrame:
    """PSI per feature between a reference frame and a current frame.

    Returns:
        Frame with feature, psi, rating -- sorted worst first.
    """
    rows = []
    for feature in features:
        value = population_stability_index(reference[feature], current[feature], bins)
        rows.append({"feature": feature, "psi": value, "rating": rate_psi(value)})
        if value >= PSI_SIGNIFICANT:
            logger.warning("feature %s: PSI %.3f (significant shift)", feature, value)
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
