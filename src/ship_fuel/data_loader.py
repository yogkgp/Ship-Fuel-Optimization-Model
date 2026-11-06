"""Source loading and explicit schema validation."""
import json

import pandas as pd
import yaml

from ship_fuel import ROOT

TARGET = "Consumer_Total_MomentaryFuel"
SPEED = "Ship_SpeedOverGround"
ENVIRONMENT = [
    "Weather_WindSpeed10M", "Weather_WaveHeight", "Weather_WavePeriod",
    "Weather_OceanCurrentVelocity", "Environment_SeaFloorDepth",
    "Weather_Temperature2M", "Ship_Bearing", "Weather_WindDirection10M",
    "Weather_WaveDirection", "Weather_OceanCurrentDirection",
]
UNITS = {SPEED: "m/s", TARGET: "kg/s", "index": "5-minute sample index",
         "Weather_WindSpeed10M": "m/s", "Weather_WaveHeight": "m",
         "Weather_WavePeriod": "s", "Weather_OceanCurrentVelocity": "m/s",
         "Environment_SeaFloorDepth": "m", "Weather_Temperature2M": "deg C",
         **{c: "degrees" for c in ENVIRONMENT if "Direction" in c or c == "Ship_Bearing"}}


def config() -> dict:
    """Read the shared project settings."""
    return yaml.safe_load((ROOT / "configs/config.yaml").read_text())


def vessel_path(vessel: str) -> "Path":
    """Resolve a configured vessel name without accepting arbitrary paths."""
    settings = config()
    if vessel not in settings["vessels"]:
        raise ValueError(f"Unknown vessel: {vessel}")
    return ROOT / "data/raw" / settings["vessels"][vessel]


def load_raw(path=None, vessel=None) -> pd.DataFrame:
    """Read untouched source parquet, rejecting incomplete schemas."""
    if path is not None and vessel is not None:
        raise ValueError("Provide path or vessel, not both")
    path = path or (vessel_path(vessel) if vessel else ROOT / "data/raw/CPS_Triton.parquet")
    frame = pd.read_parquet(path, engine="pyarrow")
    missing = set([TARGET, SPEED, "index"] + ENVIRONMENT) - set(frame)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    return frame


def save_json(path, value) -> None:
    """Write readable JSON without non-standard NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
