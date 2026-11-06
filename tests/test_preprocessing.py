import numpy as np
import pytest

from ship_fuel.data_loader import load_raw, SPEED, TARGET
from ship_fuel.preprocessing import clean, chronological_split


def test_loader_rejects_missing_schema(tmp_path):
    import pandas as pd
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"unrelated": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="Missing source columns"):
        load_raw(path)


def test_unknown_vessel_rejected():
    with pytest.raises(ValueError, match="Unknown vessel"):
        load_raw(vessel="Unknown ship")


def test_clean_audits_invalid_rows_and_retains_difficult_weather(source):
    source.loc[0, "index"] = np.nan
    source.loc[1, TARGET] = -1
    source.loc[2, SPEED] = np.nan
    source.loc[3, "Weather_WindSpeed10M"] = -3
    source.loc[4, "Weather_WaveHeight"] = 9.0
    cleaned, audit = clean(source.sample(frac=1, random_state=42))
    assert len(cleaned) == len(source) - 3
    assert cleaned["index"].is_monotonic_increasing
    assert audit["removed"]["invalid_time_index"] == 1
    assert audit["removed"]["invalid_target"] == 1
    assert audit["removed"]["invalid_ground_speed"] == 1
    assert cleaned["Weather_WindSpeed10M"].isna().sum() == 1
    assert cleaned["Weather_WaveHeight"].max() == 9


def test_time_separation(source):
    cleaned, _ = clean(source)
    splits = chronological_split(cleaned, {"train_fraction": 0.6, "validation_end": 0.8, "gap_samples": 12})
    assert splits["validation"]["index"].min() - splits["train"]["index"].max() >= 12
    assert splits["test"]["index"].min() - splits["validation"]["index"].max() >= 12
    assert set(splits["train"]["index"]).isdisjoint(splits["test"]["index"])


def test_conflicting_indices_fail(source):
    source.loc[1, "index"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        clean(source)
