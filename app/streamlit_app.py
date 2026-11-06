"""Maritime engineering demonstration using saved models and executed results."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ship_fuel.data_loader import SPEED, TARGET, UNITS
from ship_fuel.optimize import KNOT_MS, compare_speeds
from ship_fuel.predict import load_bundle, predict, predict_interval

st.set_page_config(page_title="Vessel Fuel Performance Lab", page_icon="⚓", layout="wide")
st.markdown("""<style>
:root {--navy:#102a43;--ink:#243b53;--muted:#627d98;--teal:#147d82;--paper:#f6f8fa;--line:#d9e2ec;}
.stApp {background:var(--paper);color:var(--ink);}
.block-container {max-width:1440px;padding-top:2.2rem;padding-bottom:4rem;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0b2338 0%,#102f49 100%);}
[data-testid="stSidebar"] * {color:#edf4f8;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#183b56;border-color:#486581;}
h1,h2,h3 {color:var(--navy);letter-spacing:-.025em;}
h1 {font-size:2.25rem!important;margin-bottom:.35rem!important;}
h2 {font-size:1.55rem!important;margin-top:1.4rem!important;}
h3 {font-size:1.08rem!important;}
p,li {line-height:1.62;}
[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-radius:10px;
padding:17px 18px;box-shadow:0 1px 2px rgba(16,42,67,.04);}
[data-testid="stMetricLabel"] {color:var(--muted);font-weight:600;}
[data-testid="stMetricValue"] {color:var(--navy);font-size:1.65rem;}
.page-kicker {font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:var(--teal);font-weight:750;margin-bottom:.35rem;}
.page-intro {color:var(--muted);font-size:1.02rem;max-width:850px;margin:0 0 1.6rem 0;}
.hero {background:linear-gradient(120deg,#102a43,#174b67);color:white;padding:28px 32px;border-radius:12px;margin:8px 0 24px;box-shadow:0 8px 24px rgba(16,42,67,.10);}
.hero h2 {color:white;margin:0 0 8px!important;font-size:1.65rem!important;}
.hero p {color:#d9e8ef;max-width:900px;margin:0;}
.note {background:#fff;border-left:4px solid var(--teal);padding:14px 17px;border-radius:0 8px 8px 0;margin:10px 0 18px;color:#334e68;}
.small-label {font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700;}
div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:9px;overflow:hidden;}
.stButton>button,.stDownloadButton>button {border-radius:7px;border:1px solid #9fb3c8;font-weight:600;}
hr {border-color:rgba(255,255,255,.15)!important;}
</style>""", unsafe_allow_html=True)


VESSELS = {"CPS Triton": "cps_triton", "CPS Poseidon": "cps_poseidon", "OSS Ceto": "oss_ceto"}


@st.cache_resource
def artifacts(vessel_slug):
    model_path = ROOT / "models" / vessel_slug
    return load_bundle(model_path / "selected.joblib"), json.loads((model_path / "metadata.json").read_text())


@st.cache_data
def read_data(vessel_slug, name="clean"):
    return pd.read_parquet(ROOT / f"data/processed/{vessel_slug}/{name}.parquet")


@st.cache_data
def read_metrics(vessel_slug, name):
    return pd.read_csv(ROOT / f"reports/metrics/{vessel_slug}/{name}.csv")


def chart(fig):
    fig.update_layout(template="plotly_white", font=dict(family="Arial", color="#122b42"),
                      margin=dict(l=15, r=15, t=55, b=25), colorway=["#147d82", "#d18b35", "#486581"],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                      title_font=dict(size=17, color="#102a43"), legend_title_text="")
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def scenario_controls(key, vessel_slug, vessel_name):
    """Begin with a real observed scenario; allow supported environmental edits."""
    examples = read_data(vessel_slug, "scenarios")
    label = st.selectbox("Starting conditions", examples.scenario_label.tolist(),
                         index=1, key=f"{key}_scenario")
    raw = examples.loc[examples.scenario_label.eq(label)].iloc[[0]].copy()
    st.caption(f"Based on observation {int(raw['index'].iloc[0]):,} from {vessel_name}. Weather fields are interpolated source data.")
    with st.expander("Edit weather and operating conditions", expanded=False):
        st.caption("Values begin from a real observation. The app will flag combinations that are too far from the training data.")
        cols = st.columns(2)
        definitions = [("Weather_WindSpeed10M", "Wind speed (m/s)", 0., 30.),
                       ("Weather_WaveHeight", "Wave height (m)", 0., 10.),
                       ("Weather_WavePeriod", "Wave period (s)", 0., 30.),
                       ("Weather_OceanCurrentVelocity", "Current speed (m/s)", 0., 3.),
                       ("Weather_Temperature2M", "Air temperature (°C)", -20., 50.),
                       ("Environment_SeaFloorDepth", "Sea-floor depth (m)", 0., 1000.),
                       ("Ship_Bearing", "Track bearing (degrees)", 0., 360.),
                       ("Weather_WindDirection10M", "Source wind direction (degrees)", 0., 360.),
                       ("Weather_WaveDirection", "Source wave direction (degrees)", 0., 360.),
                       ("Weather_OceanCurrentDirection", "Source current direction (degrees)", 0., 360.)]
        for i, (field, title, low, high) in enumerate(definitions):
            raw[field] = cols[i % 2].number_input(title, min_value=low, max_value=high,
                value=float(raw[field].iloc[0]), key=f"{key}_{label}_{field}")
    return raw


st.sidebar.markdown("## ⚓ Vessel Fuel Lab")
st.sidebar.caption("M.Tech research dashboard")
page = st.sidebar.radio("Workspace", ["Overview", "Data Explorer", "Fuel Prediction",
                                      "Speed Optimization", "Model Evaluation", "About the Project"])
vessel_name = st.sidebar.selectbox("Active vessel", list(VESSELS))
vessel_slug = VESSELS[vessel_name]
st.sidebar.markdown("---")
st.sidebar.caption("FuelCast operational data\n\nVessel-specific models · local CPU inference")
if not (ROOT / "models" / vessel_slug / "selected.joblib").exists():
    st.info("Train the vessel models first: .\\.venv\\Scripts\\python.exe scripts/run_multi_vessel.py")
    st.stop()
bundle, metadata = artifacts(vessel_slug)
comparison = read_metrics(vessel_slug, "model_comparison")
winner = comparison.loc[comparison.model.eq(bundle["name"]) & comparison.partition.eq("test")].iloc[0]
st.markdown(f'<div class="page-kicker">Fuel performance study · {vessel_name.upper()}</div>', unsafe_allow_html=True)
st.title(page)

if page == "Overview":
    st.markdown('''<div class="hero"><h2>Fuel prediction grounded in recorded vessel operation</h2>
    <p>Review the data, test a supported operating case, and compare voyage speeds under a stated arrival-time limit.</p></div>''', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Source records", f"{metadata['source_rows']:,}")
    b.metric("Test MAE", f"{winner.MAE_kg_s:.4f} kg/s", help="Mean absolute error on the final unseen time block")
    c.metric("Test R²", f"{winner.R2:.3f}", help="Share of test-set variation explained by the selected model")
    d.metric("Candidate models", str(comparison.model.nunique()))
    st.subheader("Study workflow")
    st.graphviz_chart('digraph {rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f3f5", color="#087e8b"]; "Raw vessel data" -> "Quality checks" -> "Time-based splits" -> "Model selection" -> "Supported speed grid" -> "Voyage comparison";}')
    left, right = st.columns([1.4, 1])
    with left:
        pred = read_metrics(vessel_slug, "test_predictions")
        chart(px.scatter(pred.iloc[::3], x="actual", y="predicted", opacity=0.35,
                         labels={"actual": "Measured fuel flow (kg/s)", "predicted": "Predicted fuel flow (kg/s)"},
                         title="Measured and predicted fuel flow on the final test period"))
    with right:
        st.subheader("How to read this dashboard")
        st.write("The model estimates total recorded fuel flow from speed over ground and available environmental context. A cubic speed curve provides the engineering baseline.")
        st.write(f"For {vessel_name}, the validation period selected **{bundle['name']}**. The final test period remained untouched until model selection was complete.")
        st.markdown('<div class="note"><b>Scope:</b> voyage figures are model-based comparisons, not measured sea-trial savings.</div>', unsafe_allow_html=True)

elif page == "Data Explorer":
    data = read_data(vessel_slug)
    st.markdown('<p class="page-intro">Explore the cleaned observations by time and broad speed regime. Filters affect every figure and summary below.</p>', unsafe_allow_html=True)
    period = st.slider("Recording period (days from the first observation)", float(data.elapsed_days.min()),
                       float(data.elapsed_days.max()), (float(data.elapsed_days.min()), float(data.elapsed_days.max())))
    regimes = st.multiselect("Speed-regime proxies", ["Near stationary", "Low speed", "Transit"],
                            default=["Near stationary", "Low speed", "Transit"])
    subset = data.loc[data.elapsed_days.between(*period) & data.regime.isin(regimes)]
    st.caption("Day 0 is the first available observation. Calendar dates are not published. Regimes are speed-based groupings, not confirmed voyage labels.")
    if subset.empty:
        st.info("No observations match the filters.")
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Records in view", f"{len(subset):,}")
    b.metric("Median speed over ground", f"{subset[SPEED].median()/KNOT_MS:.2f} kn")
    c.metric("Median fuel flow", f"{subset[TARGET].median():.4f} kg/s")
    x = st.selectbox("Horizontal axis", [SPEED, "Weather_WaveHeight", "Weather_WindSpeed10M", "Weather_OceanCurrentVelocity"],
                     help="Choose the variable to compare with measured fuel flow")
    sample = subset.iloc[::max(1, len(subset)//6000)]
    chart(px.scatter(sample, x=x, y=TARGET, color="regime", opacity=0.35,
                     labels={x: f"{x} ({UNITS[x]})", TARGET: "Measured fuel flow (kg/s)"}))
    chart(px.line(subset.iloc[::6], x="elapsed_days", y=TARGET,
                  labels={"elapsed_days": "Elapsed sequence days", TARGET: "Measured fuel flow (kg/s)"}))
    st.caption("These plots show association, not cause and effect. Route, season and operating mode may influence the same pattern.")
    with st.expander("Cleaning summary"):
        st.json(json.loads((ROOT / f"reports/metrics/{vessel_slug}/data_quality.json").read_text()))
    st.download_button("Download filtered view (CSV)", subset.to_csv(index=False), "filtered_observations.csv", "text/csv")

elif page == "Fuel Prediction":
    st.markdown('<p class="page-intro">Estimate instantaneous total fuel flow for one operating case. Start from a recorded scenario, then adjust only what you need.</p>', unsafe_allow_html=True)
    raw = scenario_controls("prediction", vessel_slug, vessel_name)
    low, high = metadata["training_ranges"][SPEED]
    speed = st.slider("Speed over ground (knots)", float(low/KNOT_MS), float(high/KNOT_MS),
                      float(np.clip(raw[SPEED].iloc[0], low, high)/KNOT_MS))
    raw[SPEED] = speed * KNOT_MS
    try:
        values, lower, upper, radius_values, _ = predict_interval(bundle, raw)
        rate = float(values[0])
        radius = float(radius_values[0])
        a, b, c = st.columns(3)
        a.metric("Estimated fuel flow", f"{rate:.4f} kg/s")
        b.metric("Hourly equivalent", f"{rate*3.6:.3f} t/h")
        c.metric("90% conformal interval", f"{lower[0]:.4f}–{upper[0]:.4f} kg/s",
                 help="Validation-calibrated prediction band; observed coverage can change in later periods")
        st.success("This case is sufficiently close to conditions represented in the training data.")
        st.caption(f"Model: {bundle['name']}. Regime-calibrated radius: ±{radius:.4f} kg/s. "
                   "The nominal 90% interval assumes calibration and future observations are exchangeable; temporal shift can change coverage.")
    except ValueError as error:
        st.warning(str(error))
    st.info("Fuel flow covers all recorded consumers. Draft, engine power and apparent wind are not used in this interactive case because they are incomplete or may reveal downstream engine response.")

elif page == "Speed Optimization":
    st.markdown('<p class="page-intro">Compare supported constant-speed cases for a fixed voyage distance and arrival limit. This is a screening calculation, not an onboard control recommendation.</p>', unsafe_allow_html=True)
    raw = scenario_controls("optimization", vessel_slug, vessel_name)
    a, b, c = st.columns(3)
    distance = a.number_input("Voyage distance (nautical miles)", min_value=1., max_value=5000., value=100.)
    max_hours = b.number_input("Maximum voyage time (hours)", min_value=0.5, max_value=500., value=14.)
    low, high = bundle["speed_bounds"]
    selected = c.number_input("Selected ground speed (knots)", min_value=low/KNOT_MS,
                              max_value=high/KNOT_MS, value=float(np.clip(raw[SPEED].iloc[0], low, high)/KNOT_MS))
    bounds = st.slider("Speeds to evaluate (knots)", low/KNOT_MS, high/KNOT_MS, (low/KNOT_MS, high/KNOT_MS))
    try:
        result = compare_speeds(bundle, raw, distance, max_hours, selected*KNOT_MS,
                                bounds[0]*KNOT_MS, bounds[1]*KNOT_MS)
        curve = result["curve"]
        fig = go.Figure()
        visible = curve.where(curve.supported)
        fig.add_scatter(x=visible.speed_kn, y=visible.fuel_kg/1000, name="Supported prediction", mode="lines")
        feasible = curve.loc[curve.feasible]
        fig.add_scatter(x=feasible.speed_kn, y=feasible.fuel_kg/1000, name="Meets arrival limit", mode="markers", marker_size=5)
        fig.update_layout(xaxis_title="Speed over ground (knots)", yaxis_title="Calculated voyage fuel (tonnes)")
        chart(fig)
        if result["best"] is None:
            st.warning(result["message"])
        else:
            best, current = result["best"], result["selected"]
            a, b, c, d = st.columns(4)
            a.metric("Lowest-fuel supported speed", f"{best['speed_kn']:.2f} kn")
            b.metric("Calculated voyage fuel", f"{best['fuel_kg']/1000:.2f} t")
            c.metric("Calculated travel time", f"{best['hours']:.2f} h")
            savings = result["savings_percent"]
            d.metric("Change from selected speed", f"{savings:.2f}% lower" if savings is not None else "Unavailable")
            if not current["supported"]:
                st.warning("Selected reference speed is unsupported; reference fuel and savings are not reported.")
            else:
                st.write(f"Selected speed: **{current['fuel_kg']/1000:.2f} t**, **{current['hours']:.2f} h**. "
                         f"Arrival limit satisfied: **{'Yes' if current['feasible'] else 'No'}**.")
                if savings is not None and savings < 0:
                    st.info("The feasible recommendation uses more predicted fuel than the reference, which may miss the arrival limit.")
            uncertainty = bundle["validation_abs_error_p90"] * best["hours"] * 3600 / 1000
            st.caption(f"Rate-error sensitivity carried over the recommended voyage: ±{uncertainty:.2f} t. "
                       "This assumes a persistent rate error; it is not a calibrated voyage confidence interval.")
        st.download_button("Download evaluated speed grid", curve.to_csv(index=False), "speed_comparison.csv", "text/csv")
        with st.expander("View all evaluated speeds"):
            st.dataframe(curve, hide_index=True)
    except ValueError as error:
        st.warning(str(error))
    st.warning("Assumptions: fixed environment, constant speed over ground, and no acceleration, routing or reserve margin. Unsupported points are excluded. The lowest grid value is not proof of a vessel's physical optimum.")

elif page == "Model Evaluation":
    st.markdown('<p class="page-intro">Inspect model selection, final-period errors and performance stability across later time windows.</p>', unsafe_allow_html=True)
    st.write(f"**Model selected before final testing:** {bundle['name']} · criterion: lowest validation MAE")
    partition = st.radio("Evaluation partition", ["test", "validation"], horizontal=True)
    table = comparison.loc[comparison.partition.eq(partition)]
    st.dataframe(table, hide_index=True, width="stretch")
    chart(px.bar(table, x="model", y="MAE_kg_s", color="model", labels={"MAE_kg_s": "MAE (kg/s)"}))
    predictions = read_metrics(vessel_slug, "test_predictions")
    left, right = st.columns(2)
    with left:
        chart(px.scatter(predictions.iloc[::max(1, len(predictions)//4000)], x="actual", y="predicted",
                         opacity=.35, title=f"{vessel_name}: measured versus predicted",
                         labels={"actual": "Measured kg/s", "predicted": "Predicted kg/s"}))
    with right:
        chart(px.histogram(predictions, x="residual", nbins=60, title="Final-test residuals",
                           labels={"residual": "Measured minus predicted (kg/s)"}))
    importance = read_metrics(vessel_slug, "permutation_importance")
    chart(px.bar(importance.head(12), x="MAE_increase_kg_s", y="feature", orientation="h",
                 title="Permutation importance on final test", labels={"MAE_increase_kg_s": "MAE increase (kg/s)"}))
    st.dataframe(read_metrics(vessel_slug, "condition_errors"), hide_index=True, width="stretch")
    st.subheader("Prediction-band coverage")
    coverage = read_metrics(vessel_slug, "uncertainty_coverage")
    st.dataframe(coverage, hide_index=True, width="stretch")
    chart(px.bar(coverage, x="group", y="observed_coverage", text_auto=".1%",
                 labels={"observed_coverage": "Observed final-test coverage"}, range_y=[0, 1]))
    st.caption("Intervals use absolute residuals from the validation region, with separate radii for speed regimes when at least 100 calibration observations are available. Coverage is diagnostic rather than guaranteed under time shift.")
    st.subheader("Performance across later time windows")
    repeated = pd.read_csv(ROOT / "reports/metrics/repeated_temporal_summary.csv")
    vessel_repeated = repeated.loc[repeated.vessel.eq(metadata["vessel"])]
    st.dataframe(vessel_repeated, hide_index=True, width="stretch")
    chart(px.bar(vessel_repeated, x="model", y="MAE_mean", error_y="MAE_std", color="model",
                 labels={"MAE_mean": "Mean MAE across 3 windows (kg/s)", "MAE_std": "MAE standard deviation"}))
    st.caption("Feature importance is predictive association and can be unstable with correlated inputs. "
               "These are separate vessel-specific models, not cross-vessel transfer models. Preprocessing was fitted only on each training window.")
    st.caption(f"Observed final-test coverage of the empirical validation error band: "
               f"{metadata['test_fraction_inside_empirical_error_band']:.1%}. No nominal coverage guarantee under temporal shift.")

else:
    st.subheader("Development of a Machine Learning-Based Decision Support System for Ship Fuel Consumption Prediction and Speed Optimization")
    st.caption("M.Tech research project · Ocean Engineering and Naval Architecture")
    st.markdown("""This study evaluates whether operational data can support useful fuel-flow estimates on later,
    unseen periods. It also demonstrates a constrained comparison of constant voyage speeds. A cubic speed curve,
    Random Forest and XGBoost are evaluated with chronological data partitions and one-day gaps between partitions.
    Model selection is completed on validation data before the final test period is scored.

    **Dataset:** anonymised CPS Triton, CPS Poseidon and OSS Ceto, FuelCast, KROHNE Digital. Source measurements include all fuel consumers;
    environmental variables are externally derived and interpolated. Only the sequence index is published.

    **Study boundary:** the models are vessel-specific and have not been verified by sea trial. Speed-through-water is
    inconsistent across the source files, while draft, hotel load and route context are incomplete. Local data-support
    checks reduce extrapolation but cannot establish physical validity. Historical weather fields are not operational forecasts.

    **Future work:** leave-one-vessel-out transfer tests, forecast uncertainty, load-aware causal validation and sea trials.

    **References:** [FuelCast dataset](https://huggingface.co/datasets/krohnedigital/FuelCast),
    [Viga et al., FuelCast paper](https://arxiv.org/abs/2510.08217).
    Source licence: CC BY-NC-ND 4.0; retain attribution and consult the licence before redistribution.
    """)
