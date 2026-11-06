"""Transparent constant-SOG voyage comparison with hard support constraints."""
import numpy as np
import pandas as pd

from ship_fuel.data_loader import SPEED
from ship_fuel.predict import predict

KNOT_MS = 1852 / 3600


def voyage(distance_nm: float, speed_ms):
    """Return seconds; nautical mile is exactly 1852 m."""
    if not np.isfinite(distance_nm) or distance_nm <= 0 or np.any(np.asarray(speed_ms) <= 0):
        raise ValueError("Distance and speed must be positive")
    return distance_nm * 1852 / np.asarray(speed_ms)


def total_fuel(rate_kg_s, duration_s):
    """Integrate an assumed constant kg/s rate over seconds to kilograms."""
    if np.any(np.asarray(rate_kg_s) < 0) or np.any(np.asarray(duration_s) < 0):
        raise ValueError("Rate and duration cannot be negative")
    return np.asarray(rate_kg_s) * np.asarray(duration_s)


def compare_speeds(bundle, scenario, distance_nm, max_hours, selected_ms,
                   minimum_ms=None, maximum_ms=None, points=151):
    """Minimize predicted voyage mass on a feasible supported finite grid.

    This is a conditional model comparison, not a validated causal control law.
    Always include the user's selected speed to prevent grid-induced false gains.
    """
    if not np.isfinite(max_hours) or max_hours <= 0:
        raise ValueError("Maximum travel time must be positive")
    low, high = bundle["speed_bounds"]
    low = max(low, minimum_ms if minimum_ms is not None else low)
    high = min(high, maximum_ms if maximum_ms is not None else high)
    if low > high:
        raise ValueError("Speed bounds do not overlap training envelope")
    if not np.isfinite(selected_ms) or not low <= selected_ms <= high:
        raise ValueError("Selected speed must lie inside the comparison bounds")
    speeds = np.unique(np.r_[np.linspace(low, high, points), selected_ms])
    raw = pd.concat([scenario.iloc[[0]]] * len(speeds), ignore_index=True)
    raw[SPEED] = speeds
    rates, supported = predict(bundle, raw, require_support=False)
    seconds = voyage(distance_nm, speeds)
    curve = pd.DataFrame({"speed_ms": speeds, "speed_kn": speeds / KNOT_MS,
                          "rate_kg_s": rates, "hours": seconds / 3600,
                          "fuel_kg": total_fuel(rates, seconds), "supported": supported})
    curve["feasible"] = curve.supported & curve.hours.le(max_hours + 1e-10)
    selected = curve.loc[curve.speed_ms.eq(selected_ms)].iloc[0]
    result = {"curve": curve, "selected": selected.to_dict(), "best": None,
              "savings_percent": None, "message": "No supported speed meets the travel-time limit."}
    if curve.feasible.any():
        best = curve.loc[curve.loc[curve.feasible, "fuel_kg"].idxmin()]
        result["best"] = best.to_dict()
        result["message"] = "Exploratory minimum among supported grid points."
        if bool(selected.supported) and selected.fuel_kg > 0:
            result["savings_percent"] = float(100 * (selected.fuel_kg - best.fuel_kg) / selected.fuel_kg)
    return result
