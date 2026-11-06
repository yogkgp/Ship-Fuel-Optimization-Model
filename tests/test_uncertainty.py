import numpy as np
import pytest

from ship_fuel.uncertainty import conformal_quantile, coverage_table, fit_calibration, interval


def test_finite_sample_quantile_and_bounds():
    scores = np.arange(1, 11)
    # ceil(11 * .9) = rank 10.
    assert conformal_quantile(scores, alpha=.1) == 10
    calibration = fit_calibration(np.zeros(300), np.ones(300),
                                  np.r_[np.zeros(100), np.ones(100), np.full(100, 3.)])
    low, high, radius = interval([.5, 2.], [0., 3.], calibration)
    assert low[0] == 0
    assert np.all(high >= np.array([.5, 2.]))
    assert np.all(radius > 0)


def test_invalid_alpha_and_coverage_shape():
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([1, 2], alpha=1)
    calibration = fit_calibration([1, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3], minimum_group=10)
    table = coverage_table([1, 2], [1, 3], [0, 3], calibration)
    assert table.iloc[0]["group"] == "Overall"
    assert table.iloc[0]["n"] == 2
