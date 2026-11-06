"""Deterministic physical cleaning, regime proxies and temporal partitions."""
import numpy as np
import pandas as pd

from ship_fuel.data_loader import ENVIRONMENT, SPEED, TARGET


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Preserve extremes; drop unusable labels/time/speed, null impossible inputs.

    No quantile or target-dependent outlier removal is performed. Environmental
    nulls are imputed later inside each model's training-only pipeline.
    """
    data = frame.copy()
    audit = {"source_rows": len(data), "removed": {}, "invalid_inputs_nulled": {}}

    def retain(mask, name):
        nonlocal data
        audit["removed"][name] = int((~mask).sum())
        data = data.loc[mask].copy()

    retain(~data.duplicated(), "exact_duplicates")
    for column in ["index", TARGET, SPEED] + ENVIRONMENT:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    retain(data["index"].notna() & data["index"].ge(0) & data["index"].mod(1).eq(0), "invalid_time_index")
    if data["index"].duplicated().any():
        raise ValueError("Conflicting sequence indices require explicit source investigation")
    retain(data[TARGET].notna() & data[TARGET].ge(0), "invalid_target")
    retain(data[SPEED].notna() & data[SPEED].ge(0), "invalid_ground_speed")
    for column in ENVIRONMENT:
        invalid = pd.Series(False, index=data.index)
        if column not in ["Weather_Temperature2M"]:
            invalid |= data[column].lt(0)
        if "Direction" in column or column == "Ship_Bearing":
            invalid |= data[column].gt(360)
        audit["invalid_inputs_nulled"][column] = int(invalid.sum())
        data.loc[invalid, column] = np.nan
    data = data.sort_values("index").reset_index(drop=True)
    data["elapsed_days"] = data["index"] * 300 / 86400
    data["regime"] = regime(data[SPEED])
    audit["retained_rows"] = len(data)
    audit["missing_after_cleaning"] = {c: int(data[c].isna().sum()) for c in [TARGET, SPEED] + ENVIRONMENT}
    audit["regime_counts"] = {str(k): int(v) for k, v in data["regime"].value_counts().items()}
    return data, audit


def regime(speed) -> pd.Series:
    """Speed proxies only: these are not verified operational labels."""
    return pd.cut(speed, [-np.inf, 0.5, 2.0, np.inf], labels=["Near stationary", "Low speed", "Transit"])


def chronological_split(data, settings):
    """Use sequence-time boundaries and an embargo before each held-out block."""
    start, end = data["index"].min(), data["index"].max() + 1
    span = end - start
    first = start + span * settings["train_fraction"]
    second = start + span * settings["validation_end"]
    gap = settings["gap_samples"]
    splits = {
        "train": data.loc[data["index"] < first - gap].copy(),
        "validation": data.loc[(data["index"] >= first) & (data["index"] < second - gap)].copy(),
        "test": data.loc[data["index"] >= second].copy(),
    }
    if any(len(part) < 20 for part in splits.values()):
        raise ValueError("Not enough data for requested chronological split and embargo")
    return splits
