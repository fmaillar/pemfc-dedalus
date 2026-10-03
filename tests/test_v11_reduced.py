"""Tests for the minimal two-state V11 reduced model."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_reduced import (
    V11ReducedInputs,
    V11ReducedParameters,
    V11ReducedState,
    reduced_v11_rhs,
)


def _inputs(current_a: float = 26.04) -> V11ReducedInputs:
    return V11ReducedInputs(
        current_a=current_a,
        stack_air_flow_slpm=216.132,
        anode_hydrogen_pressure_pa=137325.0,
        inlet_air_temperature_k=293.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.01,
    )


def _parameters() -> V11ReducedParameters:
    return V11ReducedParameters(
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )


def test_reduced_model_evaluates_nominal_point() -> None:
    derivative, outputs = reduced_v11_rhs(
        state=V11ReducedState(313.15, 5.0),
        inputs=_inputs(),
        parameters=_parameters(),
    )

    assert outputs.stack_voltage_v > 0.0
    assert outputs.stack_temperature_k == pytest.approx(313.15)
    assert outputs.cathode_oxygen_partial_pressure_pa > 0.0
    assert 0.0 < outputs.cathode_water_activity <= 1.0
    assert 0.0 < outputs.cathode_outlet_water_mole_fraction < 1.0
    assert derivative.stack_temperature_k_s == pytest.approx(
        derivative.stack_temperature_k_s
    )
    assert derivative.membrane_mean_water_content_s == pytest.approx(
        derivative.membrane_mean_water_content_s
    )


def test_membrane_hydration_changes_voltage() -> None:
    _, dry = reduced_v11_rhs(
        state=V11ReducedState(313.15, 3.0),
        inputs=_inputs(),
        parameters=_parameters(),
    )
    _, wet = reduced_v11_rhs(
        state=V11ReducedState(313.15, 10.0),
        inputs=_inputs(),
        parameters=_parameters(),
    )

    assert wet.stack_voltage_v > dry.stack_voltage_v


def test_current_increases_faraday_water_fraction_at_fixed_airflow() -> None:
    _, low = reduced_v11_rhs(
        state=V11ReducedState(313.15, 5.0),
        inputs=_inputs(current_a=5.0),
        parameters=_parameters(),
    )
    _, high = reduced_v11_rhs(
        state=V11ReducedState(313.15, 5.0),
        inputs=_inputs(current_a=26.04),
        parameters=_parameters(),
    )

    assert (
        high.cathode_outlet_water_mole_fraction
        > low.cathode_outlet_water_mole_fraction
    )
