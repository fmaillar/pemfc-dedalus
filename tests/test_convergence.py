import pytest

from pemfc_dedalus.convergence import (
    is_time_converged,
    relative_difference,
    scalar_last,
    scalar_relative_changes,
    time_convergence_metric,
)


def _report(a=1e-6, b=2e-6, c=3e-6):
    return {
        "scalar_series": {
            "mean_c_o2": {"last": 8.0, "relative_last_step_change": a},
            "mean_eta": {"last": -0.41, "relative_last_step_change": b},
            "total_reaction_current": {
                "last": 0.0048,
                "relative_last_step_change": c,
            },
        }
    }


def test_scalar_relative_changes():
    changes = scalar_relative_changes(_report())
    assert changes["mean_c_o2"] == pytest.approx(1e-6)
    assert changes["mean_eta"] == pytest.approx(2e-6)
    assert changes["total_reaction_current"] == pytest.approx(3e-6)


def test_time_convergence_metric_uses_worst_scalar():
    assert time_convergence_metric(_report()) == pytest.approx(3e-6)


def test_time_convergence_threshold():
    assert is_time_converged(_report(), 1e-5)
    assert not is_time_converged(_report(c=2e-5), 1e-5)


def test_time_convergence_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError):
        is_time_converged(_report(), 0.0)


def test_relative_difference_and_scalar_last():
    report = _report()
    assert scalar_last(report, "mean_eta") == pytest.approx(-0.41)
    assert relative_difference(1.01, 1.0) == pytest.approx(0.01)
