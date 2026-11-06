"""Bounded CPU experiments with frozen validation selection before test scoring."""
import hashlib
import importlib.metadata
import logging
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

from ship_fuel import ROOT
from ship_fuel.data_loader import SPEED, TARGET, UNITS, config, load_raw, save_json
from ship_fuel.evaluate import condition_errors, metrics
from ship_fuel.features import FEATURES, build_features
from ship_fuel.preprocessing import chronological_split, clean
from ship_fuel.validation import fit_support

LOGGER = logging.getLogger(__name__)


def candidates(settings):
    """Five candidates provide both algorithm and environmental ablations."""
    models = {"Cubic baseline": ([SPEED], Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cubic", PolynomialFeatures(3, include_bias=False)),
        ("regressor", LinearRegression())]))}
    for label, columns in [("speed", [SPEED]), ("environment", FEATURES)]:
        rf = RandomForestRegressor(n_estimators=settings["rf_trees"],
                                   min_samples_leaf=settings["rf_min_leaf"],
                                   max_depth=settings["rf_max_depth"],
                                   random_state=settings["seed"], n_jobs=settings["threads"])
        xgb = XGBRegressor(n_estimators=settings["xgb_trees"],
                           max_depth=settings["xgb_max_depth"],
                           learning_rate=settings["xgb_learning_rate"],
                           objective="reg:squarederror", tree_method="hist",
                           subsample=0.9, colsample_bytree=0.9,
                           random_state=settings["seed"], n_jobs=settings["threads"])
        for name, estimator in [("Random Forest", rf), ("XGBoost", xgb)]:
            models[f"{name} | {label}"] = (columns, Pipeline([
                ("impute", SimpleImputer(strategy="median")), ("regressor", estimator)]))
    return models


def run_training():
    """Execute cleaning, train/validation selection, then one final test audit."""
    settings = config()
    data, audit = clean(load_raw())
    splits = chronological_split(data, settings)
    for name, part in splits.items():
        part.to_parquet(ROOT / f"data/processed/{name}.parquet", index=False)
    data.to_parquet(ROOT / "data/processed/clean.parquet", index=False)
    audit["splits"] = {name: {"n": len(part), "first_index": float(part["index"].min()),
                                    "last_index": float(part["index"].max())}
                       for name, part in splits.items()}
    audit["embargo_rows"] = len(data) - sum(len(p) for p in splits.values())
    save_json(ROOT / "reports/metrics/data_quality.json", audit)
    feature_sets = {name: build_features(part) for name, part in splits.items()}
    fitted, rows = {}, []
    for name, (columns, pipeline) in candidates(settings).items():
        LOGGER.info("Training %s", name)
        before = time.perf_counter()
        pipeline.fit(feature_sets["train"][columns], splits["train"][TARGET])
        training_s = time.perf_counter() - before
        validation = np.maximum(pipeline.predict(feature_sets["validation"][columns]), 0)
        rows.append({"model": name, "partition": "validation", "training_seconds": training_s,
                     **metrics(splits["validation"][TARGET], validation)})
        fitted[name] = (columns, pipeline)
        joblib.dump({"features": columns, "pipeline": pipeline},
                    ROOT / f"models/{name.replace(' | ', '_').replace(' ', '_')}.joblib")
    selected = min(rows, key=lambda row: row["MAE_kg_s"])["model"]
    LOGGER.info("Frozen validation winner: %s. Test evaluation starts now.", selected)
    train = splits["train"]
    columns, pipeline = fitted[selected]
    residual = np.abs(splits["validation"][TARGET].to_numpy() -
                      np.maximum(pipeline.predict(feature_sets["validation"][columns]), 0))
    radius = float(np.quantile(residual, 0.90, method="higher"))
    bundle = {"pipeline": pipeline, "features": columns, "name": selected,
              "support": fit_support(feature_sets["train"]),
              "speed_bounds": (max(settings["minimum_comparison_speed_ms"],
                                   float(train[SPEED].quantile(0.01))), float(train[SPEED].quantile(0.99))),
              "validation_abs_error_p90": radius}
    joblib.dump(bundle, ROOT / "models/selected.joblib", compress=3)
    selected_predictions = None
    for name, (columns, pipeline) in fitted.items():
        before = time.perf_counter()
        predicted = np.maximum(pipeline.predict(feature_sets["test"][columns]), 0)
        elapsed = time.perf_counter() - before
        rows.append({"model": name, "partition": "test", "inference_ms_per_row": elapsed * 1000 / len(predicted),
                     **metrics(splits["test"][TARGET], predicted)})
        if name == selected:
            selected_predictions = predicted
    pd.DataFrame(rows).to_csv(ROOT / "reports/metrics/model_comparison.csv", index=False)
    test = splits["test"]
    predictions = pd.DataFrame({"index": test["index"], "elapsed_days": test.elapsed_days,
                                "speed_ms": test[SPEED], "actual": test[TARGET],
                                "predicted": selected_predictions, "regime": test.regime,
                                "wave_height": test.Weather_WaveHeight})
    predictions["residual"] = predictions.actual - predictions.predicted
    predictions["wave_band"] = pd.cut(predictions.wave_height, [-np.inf, 0.5, 1.5, np.inf],
                                      labels=["<0.5 m", "0.5–1.5 m", ">1.5 m"])
    predictions.to_csv(ROOT / "reports/metrics/test_predictions.csv", index=False)
    condition_errors(predictions).to_csv(ROOT / "reports/metrics/condition_errors.csv", index=False)
    # Interpretation is performed only after frozen model selection, never used to tune.
    sample = feature_sets["test"][bundle["features"]].iloc[::5]
    importance = permutation_importance(bundle["pipeline"], sample, test[TARGET].iloc[::5],
                                        scoring="neg_mean_absolute_error", n_repeats=3,
                                        random_state=settings["seed"], n_jobs=1)
    pd.DataFrame({"feature": bundle["features"], "MAE_increase_kg_s": importance.importances_mean,
                  "std": importance.importances_std}).sort_values("MAE_increase_kg_s", ascending=False).to_csv(
                      ROOT / "reports/metrics/permutation_importance.csv", index=False)
    metadata = {"vessel": settings["vessel"], "selected_model": selected,
                "selection_rule": "Lowest validation MAE, frozen before test evaluation; no refit",
                "features": bundle["features"], "feature_units": UNITS,
                "engineered_angle_units": "dimensionless sine/cosine of source direction minus bearing",
                "target": TARGET, "target_units": "kg/s", "config": settings,
                "source_sha256": hashlib.sha256((ROOT / "data/raw/CPS_Triton.parquet").read_bytes()).hexdigest(),
                "python": platform.python_version(), "software": {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "streamlit"]},
                "training_ranges": {c: [float(feature_sets["train"][c].min()), float(feature_sets["train"][c].max())]
                                    for c in FEATURES}, "speed_bounds_ms": bundle["speed_bounds"],
                "validation_absolute_error_p90_kg_s": radius,
                "test_fraction_inside_empirical_error_band": float((np.abs(predictions.residual) <= radius).mean())}
    save_json(ROOT / "models/metadata.json", metadata)
    return data, splits, bundle
