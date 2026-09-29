import pytest

from pemfc_dedalus.hydrogen_feedback import (
    hydrogen_partial_pressure_feedback_factor,
)


def test_zero_exponent_disables_hydrogen_feedback_exactly():
    assert hydrogen_partial_pressure_feedback_factor(
        80000.0,
        100000.0,
        0.0,
    ) == 1.0


def test_hydrogen_feedback_decreases_with_partial_pressure():
    assert hydrogen_partial_pressure_feedback_factor(
        81000.0,
        100000.0,
        0.5,
    ) == pytest.approx(0.9)
    assert hydrogen_partial_pressure_feedback_factor(
        80000.0,
        100000.0,
        1.0,
    ) == pytest.approx(0.8)


def test_hydrogen_feedback_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        hydrogen_partial_pressure_feedback_factor(-1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        hydrogen_partial_pressure_feedback_factor(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        hydrogen_partial_pressure_feedback_factor(1.0, 1.0, -1.0)
