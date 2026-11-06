import numpy as np

from ship_fuel.data_loader import SPEED
from ship_fuel.features import build_features, FEATURES


def test_angles_wrap_and_forbidden_fields_excluded(source):
    source.loc[0, "Weather_WindDirection10M"] = 360
    source.loc[1, "Weather_WindDirection10M"] = 0
    features = build_features(source)
    assert list(features) == FEATURES
    assert not any("Fuel" in field or "Power" in field or "Rotation" in field for field in features)
    assert np.isclose(features.iloc[0]["Weather_WindDirection10M_sin"], features.iloc[1]["Weather_WindDirection10M_sin"])
    assert np.isclose(features.iloc[0]["Weather_WindDirection10M_cos"], features.iloc[1]["Weather_WindDirection10M_cos"])
    assert SPEED in features


def test_training_only_imputation():
    import pandas as pd
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median").fit(pd.DataFrame({"x": [1., 3., np.nan]}))
    values = imputer.transform(pd.DataFrame({"x": [1000., np.nan]}))
    assert values[1, 0] == 2
