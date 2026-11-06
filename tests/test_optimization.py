import numpy as np
import pytest

from ship_fuel.optimize import KNOT_MS, compare_speeds, total_fuel, voyage


def test_unit_conversions():
    assert voyage(100, 10*KNOT_MS) == pytest.approx(36000)
    assert total_fuel(0.2, 36000) == pytest.approx(7200)


def test_analytic_grid_minimum(analytic_bundle, source):
    result = compare_speeds(analytic_bundle, source.iloc[[500]], 10, 10, 6.)
    # Minimize (0.04 + .001*v^3)/v: derivative zero at cube-root(20).
    assert result["best"]["speed_ms"] == pytest.approx(np.cbrt(20), abs=0.05)
    assert result["savings_percent"] > 0
    assert result["best"]["supported"]
    assert result["best"]["feasible"]
    assert result["curve"].speed_ms.eq(6.).any()


def test_deadline_forces_higher_speed(analytic_bundle, source):
    result = compare_speeds(analytic_bundle, source.iloc[[500]], 10, 1, 6.)
    assert result["best"]["speed_ms"] >= 10*KNOT_MS
    assert result["best"]["hours"] <= 1


def test_infeasible_and_outside_bounds(analytic_bundle, source):
    result = compare_speeds(analytic_bundle, source.iloc[[500]], 100, 1, 6.)
    assert result["best"] is None
    assert result["savings_percent"] is None
    with pytest.raises(ValueError, match="Selected speed"):
        compare_speeds(analytic_bundle, source.iloc[[500]], 10, 10, 12.)
    with pytest.raises(ValueError, match="positive"):
        voyage(-10, 4)


def test_unknown_reference_does_not_report_savings(analytic_bundle, source):
    scenario = source.iloc[[500]].copy()
    scenario["Weather_WaveHeight"] = 9
    result = compare_speeds(analytic_bundle, scenario, 10, 10, 6.)
    assert not result["curve"].supported.any()
    assert result["savings_percent"] is None
