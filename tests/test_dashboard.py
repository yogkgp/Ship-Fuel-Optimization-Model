"""Streamlit AppTest catches rendering failures without a manual browser."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


@pytest.mark.parametrize("vessel", ["CPS Triton", "CPS Poseidon", "OSS Ceto"])
def test_dashboard_overview_for_each_vessel(vessel):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.selectbox[0].set_value(vessel).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert vessel.upper() in app.markdown[1].value


@pytest.mark.parametrize("vessel", ["CPS Triton", "CPS Poseidon", "OSS Ceto"])
def test_prediction_page_has_interval_for_each_vessel(vessel):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.selectbox[0].set_value(vessel).run()
    app.radio[0].set_value("Fuel Prediction").run()
    assert not app.exception
    assert any(metric.label == "90% conformal interval" for metric in app.metric)


@pytest.mark.parametrize("page", ["Data Explorer", "Fuel Prediction", "Speed Optimization",
                                  "Model Evaluation", "About the Project"])
def test_dashboard_pages_render(page):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio[0].set_value(page).run()
    assert not app.exception
