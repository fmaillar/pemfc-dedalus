"""Tests for conservative V11 cathode gas balances."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_cathode import (
    CathodeGasState,
    cathode_constant_inventory_outlet_mol_s,
    cathode_gas_rhs_per_cell,
    standard_litre_per_minute_to_mol_s,
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
