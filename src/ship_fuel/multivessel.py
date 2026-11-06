"""Vessel-specific training and repeated chronological evaluation."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ship_fuel import ROOT
from ship_fuel.data_loader import SPEED, TARGET, config, load_raw, save_json, vessel_path
from ship_fuel.evaluate import condition_errors, metrics
from ship_fuel.features import FEATURES, build_features
from ship_fuel.preprocessing import clean, chronological_split
from ship_fuel.train import candidates
from ship_fuel.validation import fit_support, support_mask
from ship_fuel.uncertainty import coverage_table, fit_calibration, interval

LOGGER = logging.getLogger(__name__)


def slug(vessel: str) -> str:
    return vessel.lower()


def repeated_splits(data: pd.DataFrame, settings: dict):
    """Return expanding-window tests with a fixed pre-test embargo."""
    start = float(data["index"].min())
    end = float(data["index"].max() + 1)
    span = end - start
    gap = settings["gap_samples"]
    width = settings["repeated_test_width"] * span
    for number, fraction in enumerate(settings["repeated_test_starts"], start=1):
        boundary = start + fraction * span
        train = data.loc[data["index"] < boundary - gap].copy()
        test = data.loc[(data["index"] >= boundary) & (data["index"] < min(boundary + width, end))].copy()
        if len(train) >= 100 and len(test) >= 20:
            yield number, train, test


def train_vessel(vessel: str, settings: dict) -> dict:
    """Train five candidates, freeze the validation winner and save vessel artifacts."""
    vessel_slug = slug(vessel)
    model_dir = ROOT / "models" / vessel_slug
    metric_dir = ROOT / "reports/metrics" / vessel_slug
    processed_dir = ROOT / "data/processed" / vessel_slug
    for directory in [model_dir, metric_dir, processed_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    data, audit = clean(load_raw(vessel=vessel))
    splits = chronological_split(data, settings)
    data.to_parquet(processed_dir / "clean.parquet", index=False)
    for name, part in splits.items():
        part.to_parquet(processed_dir / f"{name}.parquet", index=False)
    audit["vessel"] = vessel
    audit["splits"] = {name: {"n": len(part), "first_index": float(part["index"].min()),
                                      "last_index": float(part["index"].max())}
                       for name, part in splits.items()}
    audit["embargo_rows"] = len(data) - sum(map(len, splits.values()))
    save_json(metric_dir / "data_quality.json", audit)
    features = {name: build_features(part) for name, part in splits.items()}
    fitted, rows = {}, []
    for name, (columns, pipeline) in candidates(settings).items():
        before = time.perf_counter()
        pipeline.fit(features["train"][columns], splits["train"][TARGET])
        train_seconds = time.perf_counter() - before
        prediction = np.maximum(pipeline.predict(features["validation"][columns]), 0)
        rows.append({"vessel": vessel, "model": name, "partition": "validation",
                     "training_seconds": train_seconds, **metrics(splits["validation"][TARGET], prediction)})
        fitted[name] = (columns, pipeline)
    selected = min(rows, key=lambda row: row["MAE_kg_s"])["model"]
    columns, selected_pipeline = fitted[selected]
    validation_prediction = np.maximum(selected_pipeline.predict(features["validation"][columns]), 0)
    calibration = fit_calibration(splits["validation"][TARGET], validation_prediction,
                                  splits["validation"][SPEED], alpha=0.10)
    error_radius = calibration["global_radius"]
    support = fit_support(features["train"])
    bundle = {"pipeline": selected_pipeline, "features": columns, "name": selected,
              "vessel": vessel, "support": support,
              "speed_bounds": (max(settings["minimum_comparison_speed_ms"],
                                    float(splits["train"][SPEED].quantile(0.01))),
                               float(splits["train"][SPEED].quantile(0.99))),
              "validation_abs_error_p90": error_radius}
    bundle["calibration"] = calibration
    joblib.dump(bundle, model_dir / "selected.joblib", compress=3)
    selected_test_prediction = None
    for name, (model_columns, pipeline) in fitted.items():
        before = time.perf_counter()
        prediction = np.maximum(pipeline.predict(features["test"][model_columns]), 0)
        rows.append({"vessel": vessel, "model": name, "partition": "test",
                     "inference_ms_per_row": (time.perf_counter() - before) * 1000 / len(prediction),
                     **metrics(splits["test"][TARGET], prediction)})
        if name == selected:
            selected_test_prediction = prediction
    comparison = pd.DataFrame(rows)
    comparison.to_csv(metric_dir / "model_comparison.csv", index=False)
    test = splits["test"]
    prediction_frame = pd.DataFrame({"index": test["index"], "elapsed_days": test.elapsed_days,
        "speed_ms": test[SPEED], "actual": test[TARGET], "predicted": selected_test_prediction,
        "regime": test.regime, "wave_height": test.Weather_WaveHeight})
    prediction_frame["residual"] = prediction_frame.actual - prediction_frame.predicted
    lower, upper, radius = interval(selected_test_prediction, test[SPEED], calibration)
    prediction_frame["lower_90"] = lower
    prediction_frame["upper_90"] = upper
    prediction_frame["interval_radius"] = radius
    prediction_frame["covered_90"] = prediction_frame.actual.between(lower, upper)
    prediction_frame["wave_band"] = pd.cut(prediction_frame.wave_height,
        [-np.inf, 0.5, 1.5, np.inf], labels=["<0.5 m", "0.5–1.5 m", ">1.5 m"])
    prediction_frame.to_csv(metric_dir / "test_predictions.csv", index=False)
    condition_errors(prediction_frame).to_csv(metric_dir / "condition_errors.csv", index=False)
    coverage_table(test[TARGET], selected_test_prediction, test[SPEED], calibration).to_csv(
        metric_dir / "uncertainty_coverage.csv", index=False)
    sample = features["test"][columns].iloc[::max(1, len(test)//3000)]
    importance = permutation_importance(selected_pipeline, sample,
        test[TARGET].iloc[::max(1, len(test)//3000)], scoring="neg_mean_absolute_error",
        n_repeats=3, random_state=settings["seed"], n_jobs=1)
    pd.DataFrame({"feature": columns, "MAE_increase_kg_s": importance.importances_mean,
                  "std": importance.importances_std}).sort_values("MAE_increase_kg_s", ascending=False).to_csv(
                      metric_dir / "permutation_importance.csv", index=False)
    # Actual supported training examples for each vessel's dashboard.
    complete = splits["train"].dropna(subset=FEATURES[:7] + ["Ship_Bearing",
        "Weather_WindDirection10M", "Weather_WaveDirection", "Weather_OceanCurrentDirection"])
    supported, _ = support_mask(build_features(complete), support)
    complete = complete.loc[supported]
    scenarios = []
    for label, quantile in [("Lower waves", .2), ("Median waves", .5), ("Higher waves", .8)]:
        desired = complete.Weather_WaveHeight.quantile(quantile)
        row = complete.loc[[(complete.Weather_WaveHeight - desired).abs().idxmin()]].copy()
        scenarios.append(row.assign(scenario_label=label))
    pd.concat(scenarios, ignore_index=True).to_parquet(processed_dir / "scenarios.parquet", index=False)
    metadata = {"vessel": vessel, "source_file": vessel_path(vessel).name,
        "source_sha256": hashlib.sha256(vessel_path(vessel).read_bytes()).hexdigest(),
        "source_rows": len(load_raw(vessel=vessel)), "retained_rows": len(data),
        "selected_model": selected, "features": columns, "speed_bounds_ms": bundle["speed_bounds"],
        "training_ranges": {c: [float(features["train"][c].min()), float(features["train"][c].max())]
                            for c in FEATURES},
        "validation_absolute_error_p90_kg_s": error_radius,
        "calibration": calibration,
        "test_fraction_inside_empirical_error_band": float(
            (np.abs(prediction_frame.residual) <= error_radius).mean())}
    save_json(model_dir / "metadata.json", metadata)
    LOGGER.info("%s selected %s", vessel, selected)
    return {"vessel": vessel, "data": data, "splits": splits, "bundle": bundle,
            "comparison": comparison, "metadata": metadata}


def run_repeated_evaluation(vessel: str, data: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Evaluate three predefined model families on three future windows."""
    keep = ["Cubic baseline", "XGBoost | speed", "XGBoost | environment"]
    rows = []
    for fold, train, test in repeated_splits(data, settings):
        train_features, test_features = build_features(train), build_features(test)
        for name, (columns, pipeline) in candidates(settings).items():
            if name not in keep:
                continue
            before = time.perf_counter()
            pipeline.fit(train_features[columns], train[TARGET])
            predicted = np.maximum(pipeline.predict(test_features[columns]), 0)
            rows.append({"vessel": vessel, "fold": fold, "model": name,
                         "train_n": len(train), "test_n": len(test),
                         "test_start_index": float(test["index"].min()),
                         "test_end_index": float(test["index"].max()),
                         "training_seconds": time.perf_counter() - before,
                         **metrics(test[TARGET], predicted)})
    return pd.DataFrame(rows)


def run_all_vessels() -> list[dict]:
    """Run vessel models sequentially to bound laptop memory use."""
    settings = config()
    results, repeated = [], []
    for vessel in settings["vessels"]:
        result = train_vessel(vessel, settings)
        results.append(result)
        repeated.append(run_repeated_evaluation(vessel, result["data"], settings))
    combined = pd.concat([r["comparison"] for r in results], ignore_index=True)
    combined.to_csv(ROOT / "reports/metrics/all_vessels_model_comparison.csv", index=False)
    repeated_frame = pd.concat(repeated, ignore_index=True)
    repeated_frame.to_csv(ROOT / "reports/metrics/repeated_temporal_evaluation.csv", index=False)
    summary = repeated_frame.groupby(["vessel", "model"], as_index=False).agg(
        folds=("fold", "count"), MAE_mean=("MAE_kg_s", "mean"), MAE_std=("MAE_kg_s", "std"),
        RMSE_mean=("RMSE_kg_s", "mean"), R2_mean=("R2", "mean"), R2_std=("R2", "std"))
    summary.to_csv(ROOT / "reports/metrics/repeated_temporal_summary.csv", index=False)
    return results
