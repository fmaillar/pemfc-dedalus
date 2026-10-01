"""Tests for V11 purge-periodic polarization helpers."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_polarization import (
    periodic_state_error,
    periodic_state_error_component,
)
from pemfc_dedalus.v11_system import V11DynamicState


def _state(*, temperature_k: float = 313.15) -> V11DynamicState:
    return V11DynamicState(
        stack_temperature_k=temperature_k,
        membrane_mean_water_content=3.0,
        anode_hydrogen_mol=5.0e-4,
        anode_nitrogen_mol=2.0e-7,
        anode_water_vapour_mol=5.0e-6,
        cathode_oxygen_mol=2.0e-4,
        cathode_nitrogen_mol=7.5e-4,
        cathode_total_water_mol=2.0e-5,
    )


def test_periodic_state_error_is_zero_for_identical_states() -> None:
    state = _state()
    assert periodic_state_error(state, state) == pytest.approx(0.0)


def test_periodic_state_error_detects_scaled_state_change() -> None:
    before = _state()
    after = _state(temperature_k=313.25)

    error = periodic_state_error(before, after)

    assert error == pytest.approx(0.1 / 313.25)



def test_periodic_state_error_component_identifies_temperature() -> None:
    before = _state()
    after = _state(temperature_k=314.15)

    assert periodic_state_error_component(before, after) == (
        "stack_temperature_k"
    )
