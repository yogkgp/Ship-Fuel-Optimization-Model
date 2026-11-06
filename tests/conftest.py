"""Small artificial fixtures test algorithms only, never represent ship results."""
import numpy as np
import pandas as pd
import pytest

from ship_fuel.data_loader import ENVIRONMENT, SPEED, TARGET
from ship_fuel.features import build_features
from ship_fuel.validation import fit_support


@pytest.fixture
def source():
    n = 1200
    result = pd.DataFrame({"index": np.arange(n), SPEED: np.linspace(2, 8, n),
                           TARGET: np.linspace(0.05, 0.3, n)})
    for field in ENVIRONMENT:
        result[field] = 1.0
    result["Ship_Bearing"] = 0.
    return result


class AnalyticRate:
    """Known cubic rate plus hotel load, used to test a voyage optimum."""
    def predict(self, features):
        return 0.04 + 0.001 * features[SPEED].to_numpy() ** 3


@pytest.fixture
def analytic_bundle(source):
    features = build_features(source)
    return {"pipeline": AnalyticRate(), "features": [SPEED],
            "support": fit_support(features), "speed_bounds": (2., 8.)}
