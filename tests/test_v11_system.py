"""Integration tests for the coupled V11 dynamic core."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_system import (
    V11DynamicState,
    coupled_v11_predictive_rhs,
    coupled_v11_rhs,
)


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


def test_coupled_rhs_preserves_global_water_balance() -> None:
    derivative, diagnostics = coupled_v11_rhs(
        state=_nominal_state(),
        current_a=26.04,
        stack_air_flow_slpm=216.1,
        cathode_outlet_molar_flow_per_cell_mol_s=1.5e-2,
        cell_voltage_v=0.768165517,
        inlet_air_temperature_k=293.15,
        cathode_air_outlet_temperature_k=303.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.01,
        nitrogen_crossover_mol_s=1.0e-8,
        dt_regulator_s=0.01,
    )

    assert diagnostics.water_conservation_residual_mol_s == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert derivative.cathode_total_water_mol_s == pytest.approx(
        26.04 / (2.0 * 96485.33212)
        + diagnostics.cathode_water_inlet_mol_s
        - diagnostics.cathode_water_outlet_mol_s
        - diagnostics.membrane.cathode_interface_rate_mol_s
    )


def test_coupled_rhs_carries_nitrogen_crossover_into_anode_inventory() -> None:
    crossover = 2.5e-8
    derivative, _ = coupled_v11_rhs(
        state=_nominal_state(),
        current_a=10.0,
        stack_air_flow_slpm=100.0,
        cathode_outlet_molar_flow_per_cell_mol_s=7.0e-3,
        cell_voltage_v=0.8,
        inlet_air_temperature_k=293.15,
        cathode_air_outlet_temperature_k=300.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.005,
        nitrogen_crossover_mol_s=crossover,
        dt_regulator_s=0.01,
    )

    assert derivative.anode_nitrogen_mol_s == pytest.approx(crossover)


def test_cathode_phase_partition_is_algebraic_diagnostic() -> None:
    state = _nominal_state()
    derivative, diagnostics = coupled_v11_rhs(
        state=state,
        current_a=0.0,
        stack_air_flow_slpm=50.0,
        cathode_outlet_molar_flow_per_cell_mol_s=3.5e-3,
        cell_voltage_v=1.0,
        inlet_air_temperature_k=293.15,
        cathode_air_outlet_temperature_k=293.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.0,
        nitrogen_crossover_mol_s=0.0,
        dt_regulator_s=0.01,
    )

    phase = diagnostics.cathode_phase
    assert phase.vapour_mol + phase.liquid_mol == pytest.approx(
        state.cathode_total_water_mol
    )
    assert 0.0 <= phase.relative_humidity <= 1.0
    assert derivative.stack_temperature_k_s == pytest.approx(0.0)


def test_thermal_closure_remains_explicit_in_coupled_rhs() -> None:
    _, cold_out = coupled_v11_rhs(
        state=_nominal_state(),
        current_a=20.0,
        stack_air_flow_slpm=150.0,
        cathode_outlet_molar_flow_per_cell_mol_s=1.0e-2,
        cell_voltage_v=0.78,
        inlet_air_temperature_k=293.15,
        cathode_air_outlet_temperature_k=298.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.005,
        nitrogen_crossover_mol_s=0.0,
        dt_regulator_s=0.01,
    )
    _, warm_out = coupled_v11_rhs(
        state=_nominal_state(),
        current_a=20.0,
        stack_air_flow_slpm=150.0,
        cathode_outlet_molar_flow_per_cell_mol_s=1.0e-2,
        cell_voltage_v=0.78,
        inlet_air_temperature_k=293.15,
        cathode_air_outlet_temperature_k=308.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.005,
        nitrogen_crossover_mol_s=0.0,
        dt_regulator_s=0.01,
    )

    assert warm_out.thermal.air_cooling_w > cold_out.thermal.air_cooling_w
    assert warm_out.thermal.temperature_rate_k_s < (
        cold_out.thermal.temperature_rate_k_s
    )



def test_predictive_rhs_closes_cell_voltage_from_dynamic_state() -> None:
    current = 26.04
    derivative, diagnostics = coupled_v11_predictive_rhs(
        state=_nominal_state(),
        current_a=current,
        stack_air_flow_slpm=216.1,
        cathode_outlet_molar_flow_per_cell_mol_s=1.5e-2,
        inlet_air_temperature_k=293.15,
        cathode_air_outlet_temperature_k=303.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.01,
        nitrogen_crossover_mol_s=1.0e-8,
        dt_regulator_s=0.01,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )

    assert diagnostics.voltage is not None
    assert diagnostics.voltage.cell_voltage_v > 0.0
    assert diagnostics.voltage.cell_voltage_v < diagnostics.voltage.reversible_v
    assert diagnostics.thermal.heat_generation_w == pytest.approx(
        10.0 * current * (1.253 - diagnostics.voltage.cell_voltage_v)
    )
    assert derivative.stack_temperature_k_s == pytest.approx(
        diagnostics.thermal.temperature_rate_k_s
    )
    assert diagnostics.water_conservation_residual_mol_s == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
