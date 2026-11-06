"""Prediction-time features; exclude every fuel, shaft-power and engine input."""
import numpy as np
import pandas as pd

from ship_fuel.data_loader import ENVIRONMENT, SPEED

SCALARS = [SPEED] + ENVIRONMENT[:6]
DIRECTIONS = ["Weather_WindDirection10M", "Weather_WaveDirection", "Weather_OceanCurrentDirection"]
FEATURES = SCALARS + [f"{c}_{trig}" for c in DIRECTIONS for trig in ["sin", "cos"]]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Encode relative angles continuously without apparent-wind leakage.

    Direction components describe source angles relative to track. They do not
    assert a common wind/current 'from' convention or derive STW from currents.
    """
    missing = set([*SCALARS, "Ship_Bearing", *DIRECTIONS]) - set(raw)
    if missing:
        raise ValueError(f"Missing scenario fields: {sorted(missing)}")
    result = raw[SCALARS].copy().astype(float)
    for column in DIRECTIONS:
        angle = np.deg2rad(raw[column] - raw["Ship_Bearing"])
        result[f"{column}_sin"] = np.sin(angle)
        result[f"{column}_cos"] = np.cos(angle)
    return result[FEATURES]
