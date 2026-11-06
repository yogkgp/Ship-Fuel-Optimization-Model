"""Single inference path shared by the dashboard and voyage comparison."""
import joblib
import numpy as np

from ship_fuel import ROOT
from ship_fuel.features import build_features
from ship_fuel.validation import support_mask
from ship_fuel.uncertainty import interval


def load_bundle(path=None):
    """Load a locally generated trusted model artifact."""
    return joblib.load(path or ROOT / "models/selected.joblib")


def predict(bundle, raw, require_support=True):
    """Return nonnegative rate predictions and support flags; never hide OOD."""
    features = build_features(raw)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Prediction requires finite values for every scenario input")
    supported, _ = support_mask(features, bundle["support"])
    if require_support and not supported.all():
        raise ValueError("Scenario outside training support; choose an observed scenario or supported inputs")
    values = bundle["pipeline"].predict(features[bundle["features"]])
    return np.maximum(values, 0), supported


def predict_interval(bundle, raw, require_support=True):
    """Return point prediction and regime-specific split-conformal bounds."""
    values, supported = predict(bundle, raw, require_support=require_support)
    if "calibration" not in bundle:
        raise ValueError("Model bundle does not contain calibrated uncertainty")
    lower, upper, radius = interval(values, raw["Ship_SpeedOverGround"], bundle["calibration"])
    return values, lower, upper, radius, supported
