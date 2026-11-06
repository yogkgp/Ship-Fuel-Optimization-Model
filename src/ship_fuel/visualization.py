"""Reproducible thesis figures and engineering studies from executed artifacts."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ship_fuel import ROOT
from ship_fuel.data_loader import SPEED, TARGET, save_json
from ship_fuel.features import build_features
from ship_fuel.optimize import compare_speeds
from ship_fuel.validation import support_mask

FIGURES = ROOT / "reports/figures"


def save_figure(name):
    """Save a consistently styled high-resolution figure."""
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def make_figures(data, splits, bundle):
    """Generate EDA, final-test diagnostics and supported scenario studies."""
    plt.rcParams.update({"figure.figsize": (9, 5), "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.2, "font.size": 11})
    for column, name, unit in [(SPEED, "speed_distribution", "Ground speed (m/s)"),
                                (TARGET, "fuel_distribution", "Total fuel flow (kg/s)")]:
        plt.figure()
        plt.hist(data[column], bins=60, color="#087e8b")
        plt.xlabel(unit)
        plt.ylabel("Observations")
        save_figure(name)
    for column, name, label in [(SPEED, "fuel_vs_speed", "Ground speed (m/s)"),
                               ("Weather_WindSpeed10M", "fuel_vs_wind", "Wind speed (m/s)"),
                               ("Weather_WaveHeight", "fuel_vs_waves", "Significant wave height (m)")]:
        plt.figure()
        valid = data[[column, TARGET]].dropna()
        plt.hexbin(valid[column], valid[TARGET], gridsize=50, mincnt=1, cmap="viridis")
        plt.colorbar(label="Observation count")
        plt.xlabel(label)
        plt.ylabel("Measured fuel flow (kg/s)")
        save_figure(name)
    fields = [SPEED, TARGET, "Weather_WindSpeed10M", "Weather_WaveHeight",
              "Weather_OceanCurrentVelocity", "Environment_SeaFloorDepth"]
    plt.figure(figsize=(10, 7))
    correlation = data[fields].corr()
    plt.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["SOG", "Fuel flow", "Wind speed", "Wave height", "Current speed", "Depth"]
    plt.xticks(range(len(fields)), labels, rotation=40, ha="right")
    plt.yticks(range(len(fields)), labels)
    for row in range(len(fields)):
        for col in range(len(fields)):
            plt.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center")
    plt.colorbar(label="Pearson correlation; association only")
    save_figure("correlation")
    plt.figure(figsize=(10, 6))
    missing = data[fields + ["Ship_Bearing", "Weather_WaveDirection"]].isna().mean() * 100
    missing.rename(index=lambda s: s.replace("Weather_", "").replace("Ship_", "")).plot.barh(color="#087e8b")
    plt.xlabel("Missing observations after cleaning (%)")
    save_figure("missing_values")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, column, label in zip(axes, [SPEED, TARGET, "Weather_WaveHeight"],
                                 ["SOG (m/s)", "Fuel (kg/s)", "Waves (m)"]):
        ax.plot(data.elapsed_days.iloc[::6], data[column].iloc[::6], linewidth=0.6, color="#087e8b")
        ax.set_ylabel(label)
        for name in ["validation", "test"]:
            ax.axvline(splits[name].elapsed_days.min(), color="#d97706", linestyle="--")
    axes[-1].set_xlabel("Elapsed sequence time (days); no calendar dates available")
    save_figure("operating_time_series")
    wave_band = pd.cut(data.Weather_WaveHeight, [0, 0.5, 1.5, np.inf],
                       labels=["<0.5 m", "0.5–1.5 m", ">1.5 m"])
    plt.figure()
    groups = [data.loc[wave_band == band, TARGET] for band in wave_band.cat.categories]
    plt.boxplot(groups, tick_labels=list(wave_band.cat.categories), showfliers=False)
    plt.ylabel("Measured total fuel flow (kg/s)")
    plt.xlabel("Wave-height band; operating mix is not controlled")
    save_figure("fuel_by_sea_state")
    pred = pd.read_csv(ROOT / "reports/metrics/test_predictions.csv")
    plt.figure()
    plt.hexbin(pred.actual, pred.predicted, gridsize=45, mincnt=1, cmap="viridis")
    lim = [0, max(pred.actual.max(), pred.predicted.max())]
    plt.plot(lim, lim, "--", color="#d97706")
    plt.xlabel("Measured fuel flow (kg/s)")
    plt.ylabel("Predicted fuel flow (kg/s)")
    save_figure("actual_vs_predicted")
    plt.figure()
    plt.hist(pred.residual, bins=60, color="#087e8b")
    plt.xlabel("Measured minus predicted fuel flow (kg/s)")
    plt.ylabel("Held-out observations")
    save_figure("residual_distribution")
    plt.figure()
    plt.scatter(pred.speed_ms, pred.residual, s=4, alpha=0.25, color="#087e8b")
    plt.axhline(0, color="#d97706", linestyle="--")
    plt.xlabel("Ground speed (m/s)")
    plt.ylabel("Residual (kg/s)")
    save_figure("residual_vs_speed")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(pred.elapsed_days, pred.actual, label="Measured", linewidth=0.8)
    axes[0].plot(pred.elapsed_days, pred.predicted, label="Predicted", linewidth=0.8)
    axes[0].legend()
    axes[0].set_ylabel("Fuel flow (kg/s)")
    axes[1].plot(pred.elapsed_days, pred.residual, linewidth=0.7)
    axes[1].set_ylabel("Residual (kg/s)")
    axes[1].set_xlabel("Elapsed sequence days in final test block")
    save_figure("test_time_series")
    importance = pd.read_csv(ROOT / "reports/metrics/permutation_importance.csv").head(10)
    plt.figure(figsize=(11, 6))
    plt.barh(importance.feature.iloc[::-1], importance.MAE_increase_kg_s.iloc[::-1], color="#087e8b")
    plt.xlabel("Increase in held-out MAE after permutation (kg/s)")
    save_figure("feature_importance")
    errors = pd.read_csv(ROOT / "reports/metrics/condition_errors.csv")
    for grouping in ["regime", "wave_band"]:
        part = errors.loc[errors.grouping.eq(grouping)]
        plt.figure()
        plt.bar(part.condition, part.MAE_kg_s, color="#087e8b")
        plt.ylabel("Held-out MAE (kg/s)")
        plt.xlabel(f"{grouping}; sample counts in condition_errors.csv")
        save_figure(f"errors_{grouping}")
    make_studies(splits["train"], bundle)


def make_studies(train, bundle):
    """Choose actual training scenarios; no fabricated operating conditions."""
    eligible = train.loc[train[SPEED].between(*bundle["speed_bounds"])].dropna(
        subset=[*bundle["support"]["columns"][:1], "Weather_WaveHeight", "Weather_WindSpeed10M"])
    valid, _ = support_mask(build_features(eligible), bundle["support"])
    eligible = eligible.loc[valid]
    scenarios, summaries = [], []
    # Typical observed low/medium/high wave scenarios; other covariates differ too.
    for label, quantile in [("Lower waves", 0.2), ("Median waves", 0.5), ("Higher waves", 0.8)]:
        desired = eligible.Weather_WaveHeight.quantile(quantile)
        index = (eligible.Weather_WaveHeight - desired).abs().idxmin()
        scenario = eligible.loc[[index]].copy()
        speed = float(scenario[SPEED].iloc[0])
        scenarios.append(scenario.assign(scenario_label=label))
        result = compare_speeds(bundle, scenario, 100, 14, speed)
        result["curve"].assign(scenario=label).to_csv(
            ROOT / f"reports/metrics/speed_curve_{quantile}.csv", index=False)
        summaries.append({"scenario": label, "source_index": float(scenario["index"].iloc[0]),
                          "wave_m": float(scenario.Weather_WaveHeight.iloc[0]),
                          "wind_ms": float(scenario.Weather_WindSpeed10M.iloc[0]),
                          "selected": result["selected"], "best": result["best"],
                          "savings_percent": result["savings_percent"]})
    scenario_data = pd.concat(scenarios, ignore_index=True)
    scenario_data.to_parquet(ROOT / "data/processed/scenarios.parquet", index=False)
    save_json(ROOT / "reports/metrics/scenario_summary.json", summaries)
    demo = scenario_data.iloc[[1]]
    demo_speed = float(demo[SPEED].iloc[0])
    rows = []
    for hours in [6, 8, 10, 12, 14, 16, 20]:
        result = compare_speeds(bundle, demo, 100, hours, demo_speed)
        best = result["best"]
        rows.append({"max_hours": hours, "feasible": best is not None,
                     "best_speed_ms": best["speed_ms"] if best else np.nan,
                     "fuel_kg": best["fuel_kg"] if best else np.nan,
                     "hours": best["hours"] if best else np.nan})
    pd.DataFrame(rows).to_csv(ROOT / "reports/metrics/time_tradeoff.csv", index=False)
    plt.figure()
    for quantile, label in [(0.2, "Lower waves"), (0.5, "Median waves"), (0.8, "Higher waves")]:
        curve = pd.read_csv(ROOT / f"reports/metrics/speed_curve_{quantile}.csv")
        supported = curve.where(curve.supported)
        plt.plot(supported.speed_kn, supported.fuel_kg / 1000, label=label)
    plt.xlabel("Ground speed (knots)")
    plt.ylabel("Predicted fuel for 100 nm (tonnes)")
    plt.legend(title="Observed scenario; gaps = unsupported")
    save_figure("voyage_speed_comparison")
    plt.figure()
    tradeoff = pd.DataFrame(rows)
    plt.plot(tradeoff.max_hours, tradeoff.best_speed_ms, "o-", color="#087e8b")
    plt.xlabel("Maximum permitted voyage time (hours), 100 nm")
    plt.ylabel("Supported grid minimum SOG (m/s)")
    save_figure("voyage_time_tradeoff")
