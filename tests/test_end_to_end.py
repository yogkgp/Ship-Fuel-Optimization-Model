"""Integration tests require completed real-data experiments."""
import numpy as np
import pandas as pd
import pytest

from ship_fuel import ROOT
from ship_fuel.data_loader import SPEED
from ship_fuel.predict import load_bundle, predict
from ship_fuel.optimize import compare_speeds
from ship_fuel.predict import predict_interval


@pytest.mark.skipif(not (ROOT / "models/selected.joblib").exists(), reason="Run real pipeline first")
def test_real_artifacts_and_supported_demo():
    bundle = load_bundle()
    scenarios = pd.read_parquet(ROOT / "data/processed/scenarios.parquet")
    raw = scenarios.iloc[[1]]
    values, supported = predict(bundle, raw)
    assert np.isfinite(values).all() and supported.all()
    comparison = compare_speeds(bundle, raw, 100, 14, float(raw[SPEED].iloc[0]))
    assert comparison["best"] is not None
    assert comparison["best"]["hours"] <= 14


@pytest.mark.parametrize("slug", ["cps_triton", "cps_poseidon", "oss_ceto"])
def test_vessel_bundle_has_calibrated_interval(slug):
    path = ROOT / "models" / slug / "selected.joblib"
    scenarios_path = ROOT / "data/processed" / slug / "scenarios.parquet"
    if not path.exists():
        pytest.skip("Run multi-vessel pipeline first")
    bundle = load_bundle(path)
    raw = pd.read_parquet(scenarios_path).iloc[[1]]
    point, lower, upper, radius, supported = predict_interval(bundle, raw)
    assert supported.all()
    assert lower[0] <= point[0] <= upper[0]
    assert radius[0] > 0
