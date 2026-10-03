"""Tests for V11 local observability analysis."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.v11_interfaces import (
    V11ExogenousInputs,
    V11ModelParameters,
)
from pemfc_dedalus.v11_observability import local_observability_report
from pemfc_dedalus.v11_system import V11DynamicState


def _nominal_state() -> V11DynamicState:
    return V11DynamicState(
        stack_temperature_k=313.15,
        membrane_mean_water_content=5.0,
        anode_hydrogen_mol=4.5e-4,
        anode_nitrogen_mol=2.0e-5,
        anode_water_vapour_mol=1.0e-5,
        cathode_oxygen_mol=2.1e-4,
        cathode_nitrogen_mol=7.9e-4,
        cathode_total_water_mol=5.0e-5,
    )


def _exogenous() -> V11ExogenousInputs:
    return V11ExogenousInputs(
        current_a=26.04,
        inlet_air_temperature_k=293.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.01,
    )


def _parameters() -> V11ModelParameters:
    return V11ModelParameters(
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )


def test_current_measurement_has_no_direct_state_sensitivity() -> None:
    report = local_observability_report(
        state=_nominal_state(),
        current_a=26.04,
        stack_air_flow_slpm=216.1,
        exogenous=_exogenous(),
        parameters=_parameters(),
    )

    assert np.asarray(report.current_measurement_state_sensitivity) == pytest.approx(
        np.zeros(report.state_dimension),
        abs=1.0e-12,
    )


def test_temperature_measurement_directly_exposes_thermal_state() -> None:
    state = _nominal_state()
    report = local_observability_report(
        state=state,
        current_a=26.04,
        stack_air_flow_slpm=216.1,
        exogenous=_exogenous(),
        parameters=_parameters(),
    )

    sensitivity = np.asarray(report.temperature_state_sensitivity)
    assert sensitivity[0] == pytest.approx(state.stack_temperature_k)
    assert sensitivity[1:] == pytest.approx(np.zeros(7), abs=1.0e-12)


def test_observability_report_has_consistent_dimensions() -> None:
    report = local_observability_report(
        state=_nominal_state(),
        current_a=26.04,
        stack_air_flow_slpm=216.1,
        exogenous=_exogenous(),
        parameters=_parameters(),
    )

    assert report.state_dimension == 8
    assert len(report.state_names) == 8
    assert len(report.singular_values) == 8
    assert 1 <= report.rank <= report.state_dimension
