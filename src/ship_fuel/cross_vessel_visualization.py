"""Figures comparing executed vessel-specific and repeated-window results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ship_fuel import ROOT


def make_cross_vessel_figures():
    comparison = pd.read_csv(ROOT / "reports/metrics/all_vessels_model_comparison.csv")
    test = comparison.loc[comparison.partition.eq("test")].copy()
    labels = {"CPS_Triton": "CPS Triton", "CPS_Poseidon": "CPS Poseidon", "OSS_Ceto": "OSS Ceto"}
    test["vessel_label"] = test.vessel.map(labels)
    models = test.model.unique()
    vessels = list(labels.values())
    x = np.arange(len(vessels))
    width = .16
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, model in enumerate(models):
        values = [test.loc[(test.vessel_label.eq(vessel)) & (test.model.eq(model)), "MAE_kg_s"].iloc[0]
                  for vessel in vessels]
        ax.bar(x + (offset - 2) * width, values, width, label=model)
    ax.set_xticks(x, vessels)
    ax.set_ylabel("Final-test MAE (kg/s)")
    ax.set_title("Vessel-specific model comparison on future time blocks")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/all_vessels_test_mae.png", dpi=180)
    plt.close(fig)

    repeated = pd.read_csv(ROOT / "reports/metrics/repeated_temporal_summary.csv")
    repeated["vessel_label"] = repeated.vessel.map(labels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, vessel in zip(axes, vessels):
        part = repeated.loc[repeated.vessel_label.eq(vessel)]
        ax.barh(part.model, part.MAE_mean, xerr=part.MAE_std, color=["#78909c", "#087e8b", "#e49c36"])
        ax.set_title(vessel)
        ax.set_xlabel("Mean MAE ± SD (kg/s), 3 windows")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Repeated chronological evaluation reveals temporal variability")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/repeated_temporal_mae.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    qualities = []
    for vessel, vessel_label in labels.items():
        path = ROOT / "reports/metrics" / vessel.lower() / "data_quality.json"
        import json
        audit = json.loads(path.read_text())
        qualities.append({"vessel": vessel_label, "source": audit["source_rows"],
                          "retained": audit["retained_rows"],
                          "near_stationary": audit["regime_counts"].get("Near stationary", 0),
                          "low_speed": audit["regime_counts"].get("Low speed", 0),
                          "transit": audit["regime_counts"].get("Transit", 0)})
    quality = pd.DataFrame(qualities)
    quality.to_csv(ROOT / "reports/metrics/vessel_data_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(quality))
    for column, label, color in [("near_stationary", "Near stationary", "#78909c"),
                                 ("low_speed", "Low speed", "#e49c36"),
                                 ("transit", "Transit", "#087e8b")]:
        share = 100 * quality[column] / quality.retained
        ax.bar(quality.vessel, share, bottom=bottom, label=label, color=color)
        bottom += share
    ax.set_ylabel("Retained observations (%)")
    ax.set_title("Operating-profile proxy differs strongly by vessel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/vessel_regime_comparison.png", dpi=180)
    plt.close(fig)

    coverage_rows = []
    for vessel, vessel_label in labels.items():
        coverage = pd.read_csv(ROOT / "reports/metrics" / vessel.lower() / "uncertainty_coverage.csv")
        overall = coverage.loc[coverage.group.eq("Overall")].iloc[0]
        coverage_rows.append({"vessel": vessel_label, "coverage": overall.observed_coverage,
                              "width": overall.mean_interval_width_kg_s})
    coverage_data = pd.DataFrame(coverage_rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(coverage_data.vessel, 100 * coverage_data.coverage,
                  color=["#087e8b", "#385b83", "#e49c36"])
    ax.axhline(90, color="#a83232", linestyle="--", label="Nominal 90%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Observed final-test coverage (%)")
    ax.set_title("Chronological shift reduces interval coverage for OSS Ceto")
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/uncertainty_coverage.png", dpi=180)
    plt.close(fig)
