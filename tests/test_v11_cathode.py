"""Tests for conservative V11 cathode gas balances."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_cathode import (
    CathodeGasState,
    cathode_constant_inventory_outlet_mol_s,
    cathode_gas_rhs_per_cell,
    cathode_isobaric_outlet_mol_s,
    standard_litre_per_minute_to_mol_s,
    water_saturation_pressure_derivative_pa_k,
)
from pemfc_dedalus.v11_galvanostatic import faraday_rates_per_cell


def test_slpm_conversion_matches_ideal_gas_law() -> None:
    molar = standard_litre_per_minute_to_mol_s(60.0)

    assert molar == pytest.approx(
        101325.0 * 1.0e-3 / (8.31446261815324 * 273.15)
    )


def test_constant_inventory_outlet_closes_total_moles() -> None:
    current_a = 26.04
    faraday = faraday_rates_per_cell(current_a)
    inlet = 1.0e-3
    water_to_gas = faraday.water_production_mol_s

    outlet = cathode_constant_inventory_outlet_mol_s(
        inlet_air_mol_s=inlet,
        current_a=current_a,
        water_source_to_gas_mol_s=water_to_gas,
    )

    assert outlet == pytest.approx(
        inlet - faraday.oxygen_consumption_mol_s + water_to_gas
    )


def test_cathode_rhs_obeys_total_molar_balance() -> None:
    state = CathodeGasState(
        oxygen_mol=0.21e-3,
        nitrogen_mol=0.78e-3,
        water_vapour_mol=0.01e-3,
    )
    current_a = 26.04
    stack_air_flow_slpm = 100.0
    n_cells = 10
    inlet_per_cell = standard_litre_per_minute_to_mol_s(
        stack_air_flow_slpm / n_cells
    )
    water_to_gas = 2.0e-5
    outlet = cathode_constant_inventory_outlet_mol_s(
        inlet_air_mol_s=inlet_per_cell,
        current_a=current_a,
        water_source_to_gas_mol_s=water_to_gas,
    )

    rhs = cathode_gas_rhs_per_cell(
        state=state,
        stack_air_flow_slpm=stack_air_flow_slpm,
        n_cells=n_cells,
        current_a=current_a,
        inlet_oxygen_mole_fraction=0.21,
        inlet_water_mole_fraction=0.01,
        water_source_to_gas_mol_s=water_to_gas,
        outlet_molar_flow_per_cell_mol_s=outlet,
    )

    assert rhs.total_mol_s == pytest.approx(0.0, abs=1.0e-15)


def test_zero_current_with_matching_composition_is_steady() -> None:
    state = CathodeGasState(
        oxygen_mol=0.21,
        nitrogen_mol=0.78,
        water_vapour_mol=0.01,
    )
    stack_air_flow_slpm = 100.0
    n_cells = 10
    inlet_per_cell = standard_litre_per_minute_to_mol_s(
        stack_air_flow_slpm / n_cells
    )

    rhs = cathode_gas_rhs_per_cell(
        state=state,
        stack_air_flow_slpm=stack_air_flow_slpm,
        n_cells=n_cells,
        current_a=0.0,
        inlet_oxygen_mole_fraction=0.21,
        inlet_water_mole_fraction=0.01,
        water_source_to_gas_mol_s=0.0,
        outlet_molar_flow_per_cell_mol_s=inlet_per_cell,
    )

    assert rhs.oxygen_mol_s == pytest.approx(0.0, abs=1.0e-15)
    assert rhs.nitrogen_mol_s == pytest.approx(0.0, abs=1.0e-15)
    assert rhs.water_vapour_mol_s == pytest.approx(0.0, abs=1.0e-15)


def test_positive_current_depletes_oxygen_without_airflow() -> None:
    state = CathodeGasState(
        oxygen_mol=0.21,
        nitrogen_mol=0.78,
        water_vapour_mol=0.01,
    )

    rhs = cathode_gas_rhs_per_cell(
        state=state,
        stack_air_flow_slpm=0.0,
        n_cells=10,
        current_a=10.0,
        inlet_oxygen_mole_fraction=0.21,
        inlet_water_mole_fraction=0.01,
        water_source_to_gas_mol_s=0.0,
        outlet_molar_flow_per_cell_mol_s=0.0,
    )

    assert rhs.oxygen_mol_s < 0.0
    assert rhs.nitrogen_mol_s == pytest.approx(0.0)
    assert rhs.water_vapour_mol_s == pytest.approx(0.0)



def test_cathode_water_exchange_can_be_a_signed_sink() -> None:
    state = CathodeGasState(
        oxygen_mol=0.21,
        nitrogen_mol=0.78,
        water_vapour_mol=0.01,
    )

    rhs = cathode_gas_rhs_per_cell(
        state=state,
        stack_air_flow_slpm=0.0,
        n_cells=10,
        current_a=0.0,
        inlet_oxygen_mole_fraction=0.21,
        inlet_water_mole_fraction=0.01,
        water_source_to_gas_mol_s=-2.0e-6,
        outlet_molar_flow_per_cell_mol_s=0.0,
    )

    assert rhs.water_vapour_mol_s == pytest.approx(-2.0e-6)



def test_isobaric_outlet_reduces_to_constant_inventory_at_fixed_temperature() -> None:
    state = CathodeGasState(
        oxygen_mol=0.21e-3,
        nitrogen_mol=0.78e-3,
        water_vapour_mol=0.01e-3,
    )
    inlet = 1.0e-3
    current = 26.04
    water_to_gas = 2.0e-5

    isobaric = cathode_isobaric_outlet_mol_s(
        state=state,
        liquid_water_mol=0.0,
        inlet_air_mol_s=inlet,
        inlet_water_mole_fraction=0.01,
        current_a=current,
        water_source_to_gas_mol_s=water_to_gas,
        temperature_k=313.15,
        temperature_rate_k_s=0.0,
        total_pressure_pa=101325.0,
    )
    constant_inventory = cathode_constant_inventory_outlet_mol_s(
        inlet_air_mol_s=inlet,
        current_a=current,
        water_source_to_gas_mol_s=water_to_gas,
    )

    assert isobaric == pytest.approx(constant_inventory)


def test_isobaric_outlet_accounts_for_thermal_expansion() -> None:
    state = CathodeGasState(
        oxygen_mol=0.21e-3,
        nitrogen_mol=0.78e-3,
        water_vapour_mol=0.01e-3,
    )
    temperature = 313.15
    temperature_rate = 2.0
    inlet = 1.0e-3
    water_to_gas = 2.0e-5
    current = 10.0

    fixed_temperature = cathode_isobaric_outlet_mol_s(
        state=state,
        liquid_water_mol=0.0,
        inlet_air_mol_s=inlet,
        inlet_water_mole_fraction=0.01,
        current_a=current,
        water_source_to_gas_mol_s=water_to_gas,
        temperature_k=temperature,
        temperature_rate_k_s=0.0,
        total_pressure_pa=101325.0,
    )
    heating = cathode_isobaric_outlet_mol_s(
        state=state,
        liquid_water_mol=0.0,
        inlet_air_mol_s=inlet,
        inlet_water_mole_fraction=0.01,
        current_a=current,
        water_source_to_gas_mol_s=water_to_gas,
        temperature_k=temperature,
        temperature_rate_k_s=temperature_rate,
        total_pressure_pa=101325.0,
    )

    assert heating - fixed_temperature == pytest.approx(
        state.total_mol * temperature_rate / temperature
    )


def test_buck_saturation_derivative_matches_central_difference() -> None:
    from pemfc_dedalus.anode import water_saturation_pressure_pa

    temperature = 313.15
    delta = 1.0e-3
    numerical = (
        water_saturation_pressure_pa(temperature + delta)
        - water_saturation_pressure_pa(temperature - delta)
    ) / (2.0 * delta)

    analytic = water_saturation_pressure_derivative_pa_k(temperature)

    assert analytic == pytest.approx(numerical, rel=1.0e-8)



def test_isobaric_outlet_preserves_pressure_with_saturated_vapour() -> None:
    from pemfc_dedalus.anode import water_saturation_pressure_pa

    temperature = 313.15
    pressure = 101325.0
    temperature_rate = 1.5
    dry_mol = 1.0e-3
    saturation_pressure = water_saturation_pressure_pa(temperature)
    water_fraction = saturation_pressure / pressure
    dry_fraction = 1.0 - water_fraction
    vapour_mol = water_fraction / dry_fraction * dry_mol
    state = CathodeGasState(
        oxygen_mol=0.21 * dry_mol,
        nitrogen_mol=0.79 * dry_mol,
        water_vapour_mol=vapour_mol,
    )
    inlet = 1.2e-3
    inlet_water_fraction = 0.02
    current = 20.0

    outlet = cathode_isobaric_outlet_mol_s(
        state=state,
        liquid_water_mol=2.0e-4,
        inlet_air_mol_s=inlet,
        inlet_water_mole_fraction=inlet_water_fraction,
        current_a=current,
        water_source_to_gas_mol_s=5.0e-5,
        temperature_k=temperature,
        temperature_rate_k_s=temperature_rate,
        total_pressure_pa=pressure,
    )

    faraday = faraday_rates_per_cell(current)
    dry_rate = (
        inlet * (1.0 - inlet_water_fraction)
        - outlet * dry_fraction
        - faraday.oxygen_consumption_mol_s
    )
    dp_sat_dtemperature = water_saturation_pressure_derivative_pa_k(
        temperature
    )
    dq_dtemperature = (
        pressure
        * dp_sat_dtemperature
        / (pressure - saturation_pressure) ** 2
    )
    gas_rate = (
        dry_rate / dry_fraction
        + dry_mol * dq_dtemperature * temperature_rate
    )

    assert gas_rate == pytest.approx(
        -state.total_mol * temperature_rate / temperature,
        abs=1.0e-15,
    )
