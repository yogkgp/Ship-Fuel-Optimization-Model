import joblib
import numpy as np
import pytest

from ship_fuel.data_loader import SPEED
from ship_fuel.predict import predict, load_bundle
from ship_fuel.features import build_features
from ship_fuel.validation import support_mask


def test_model_roundtrip_and_inference(analytic_bundle, source, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(analytic_bundle, path)
    values, supported = predict(load_bundle(path), source.iloc[[600]])
    assert values[0] > 0
    assert supported.all()


def test_prediction_rejects_missing_and_outside_support(analytic_bundle, source):
    bad = source.iloc[[500]].copy()
    bad[SPEED] = 80
    with pytest.raises(ValueError, match="outside training support"):
        predict(analytic_bundle, bad)
    bad[SPEED] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predict(analytic_bundle, bad)


def test_marginally_invalid_environment_rejected(analytic_bundle, source):
    bad = source.iloc[[500]].copy()
    bad["Weather_WaveHeight"] = 10
    valid, _ = support_mask(build_features(bad), analytic_bundle["support"])
    assert not valid.any()
