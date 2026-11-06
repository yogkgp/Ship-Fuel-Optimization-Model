"""Finite-sample split-conformal intervals calibrated by speed regime."""
import math

import numpy as np
import pandas as pd

from ship_fuel.preprocessing import regime


def conformal_quantile(scores, alpha=0.10) -> float:
    """Return the finite-sample conformal residual quantile.

    Exchangeability is an assumption, not guaranteed for chronological data.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    if len(values) == 0:
        raise ValueError("At least one finite calibration score is required")
    rank = min(math.ceil((len(values) + 1) * (1 - alpha)), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


def fit_calibration(actual, predicted, speed, alpha=0.10, minimum_group=100) -> dict:
    """Fit global and regime-specific absolute-residual radii."""
    frame = pd.DataFrame({"score": np.abs(np.asarray(actual) - np.asarray(predicted)),
                          "regime": regime(pd.Series(np.asarray(speed))).astype(str)})
    global_radius = conformal_quantile(frame.score, alpha)
    groups = {}
    counts = {}
    for name, part in frame.groupby("regime"):
        counts[name] = len(part)
        groups[name] = conformal_quantile(part.score, alpha) if len(part) >= minimum_group else global_radius
    return {"alpha": alpha, "nominal_coverage": 1 - alpha, "global_radius": global_radius,
            "regime_radii": groups, "calibration_counts": counts,
            "minimum_group": minimum_group,
            "warning": "Coverage assumes exchangeability; chronological distribution shift can change observed coverage."}


def radii_for_speed(calibration: dict, speed) -> np.ndarray:
    """Map each SOG to its calibrated regime radius, with global fallback."""
    labels = regime(pd.Series(np.asarray(speed))).astype(str)
    return labels.map(calibration["regime_radii"]).fillna(calibration["global_radius"]).to_numpy(float)


def interval(predicted, speed, calibration):
    """Return nonnegative lower and upper fuel-rate bounds."""
    predicted = np.asarray(predicted, dtype=float)
    radius = radii_for_speed(calibration, speed)
    return np.maximum(predicted - radius, 0), predicted + radius, radius


def coverage_table(actual, predicted, speed, calibration) -> pd.DataFrame:
    """Observed final-block coverage overall and by regime."""
    actual = np.asarray(actual, dtype=float)
    lower, upper, radius = interval(predicted, speed, calibration)
    frame = pd.DataFrame({"actual": actual, "lower": lower, "upper": upper, "radius": radius,
                          "regime": regime(pd.Series(np.asarray(speed))).astype(str)})
    frame["covered"] = frame.actual.between(frame.lower, frame.upper)
    rows = [{"group": "Overall", "n": len(frame), "observed_coverage": frame.covered.mean(),
             "mean_interval_width_kg_s": (frame.upper - frame.lower).mean()}]
    for name, part in frame.groupby("regime"):
        rows.append({"group": name, "n": len(part), "observed_coverage": part.covered.mean(),
                     "mean_interval_width_kg_s": (part.upper - part.lower).mean()})
    return pd.DataFrame(rows)
