"""Tests for the globally conservative V11 water balance."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_galvanostatic import faraday_rates_per_cell
from pemfc_dedalus.v11_water import (
    water_balance_residual_per_cell,
    water_inventory_rhs_per_cell,
)


def test_internal_transfers_cancel_from_global_water_balance() -> None:
    derivative = water_inventory_rhs_per_cell(
        current_a=0.0,
        anode_interface_rate_into_membrane_mol_s=2.0e-6,
        cathode_interface_rate_into_membrane_mol_s=-3.0e-6,
        cathode_phase_change_vapour_to_liquid_mol_s=4.0e-6,
        cathode_inlet_water_mol_s=0.0,
        cathode_outlet_water_mol_s=0.0,
        anode_purge_water_mol_s=0.0,
    )

    assert derivative.total_mol_s == pytest.approx(0.0, abs=1.0e-18)


def test_faraday_water_generation_is_only_source_in_closed_system() -> None:
    current_a = 26.04
    derivative = water_inventory_rhs_per_cell(
        current_a=current_a,
        anode_interface_rate_into_membrane_mol_s=0.0,
        cathode_interface_rate_into_membrane_mol_s=0.0,
        cathode_phase_change_vapour_to_liquid_mol_s=0.0,
        cathode_inlet_water_mol_s=0.0,
        cathode_outlet_water_mol_s=0.0,
        anode_purge_water_mol_s=0.0,
    )

    expected = faraday_rates_per_cell(current_a).water_production_mol_s

    assert derivative.cathode_vapour_mol_s == pytest.approx(expected)
    assert derivative.total_mol_s == pytest.approx(expected)


def test_phase_change_moves_water_without_creating_it() -> None:
    rate = 5.0e-6
    derivative = water_inventory_rhs_per_cell(
        current_a=0.0,
        anode_interface_rate_into_membrane_mol_s=0.0,
        cathode_interface_rate_into_membrane_mol_s=0.0,
        cathode_phase_change_vapour_to_liquid_mol_s=rate,
        cathode_inlet_water_mol_s=0.0,
        cathode_outlet_water_mol_s=0.0,
        anode_purge_water_mol_s=0.0,
    )

    assert derivative.cathode_vapour_mol_s == pytest.approx(-rate)
    assert derivative.cathode_liquid_mol_s == pytest.approx(rate)
    assert derivative.total_mol_s == pytest.approx(0.0)


def test_interface_exchange_is_pairwise_conservative() -> None:
    anode_rate = 2.0e-6
    cathode_rate = -7.0e-6
    derivative = water_inventory_rhs_per_cell(
        current_a=0.0,
        anode_interface_rate_into_membrane_mol_s=anode_rate,
        cathode_interface_rate_into_membrane_mol_s=cathode_rate,
        cathode_phase_change_vapour_to_liquid_mol_s=0.0,
        cathode_inlet_water_mol_s=0.0,
        cathode_outlet_water_mol_s=0.0,
        anode_purge_water_mol_s=0.0,
    )

    assert derivative.anode_vapour_mol_s == pytest.approx(-anode_rate)
    assert derivative.membrane_water_mol_s == pytest.approx(
        anode_rate + cathode_rate
    )
    assert derivative.cathode_vapour_mol_s == pytest.approx(-cathode_rate)
    assert derivative.total_mol_s == pytest.approx(0.0)


def test_global_water_residual_is_zero_with_external_flows() -> None:
    current_a = 10.0
    cathode_inlet = 3.0e-6
    cathode_outlet = 8.0e-6
    purge = 1.0e-6

    derivative = water_inventory_rhs_per_cell(
        current_a=current_a,
        anode_interface_rate_into_membrane_mol_s=2.0e-6,
        cathode_interface_rate_into_membrane_mol_s=-1.0e-6,
        cathode_phase_change_vapour_to_liquid_mol_s=4.0e-6,
        cathode_inlet_water_mol_s=cathode_inlet,
        cathode_outlet_water_mol_s=cathode_outlet,
        anode_purge_water_mol_s=purge,
    )
    residual = water_balance_residual_per_cell(
        derivative=derivative,
        current_a=current_a,
        cathode_inlet_water_mol_s=cathode_inlet,
        cathode_outlet_water_mol_s=cathode_outlet,
        anode_purge_water_mol_s=purge,
    )

    assert residual.actual_total_rate_mol_s == pytest.approx(
        residual.expected_total_rate_mol_s
    )
    assert residual.residual_mol_s == pytest.approx(0.0, abs=1.0e-18)
