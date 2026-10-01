"""Tests for V11 cathode water phase equilibrium."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_phase_change import (
    equilibrium_phase_change_mol,
    repartition_cathode_water_equilibrium,
    saturated_cathode_water_vapour_mol,
)


def test_saturation_inventory_matches_fixed_pressure_mole_fraction() -> None:
    dry_gas = 1.0e-3
    temperature = 313.15
    pressure = 101325.0

    water_sat = saturated_cathode_water_vapour_mol(
        dry_gas_mol=dry_gas,
        temperature_k=temperature,
        total_pressure_pa=pressure,
    )
    y_water = water_sat / (dry_gas + water_sat)

    state = repartition_cathode_water_equilibrium(
        total_water_mol=10.0 * water_sat,
        dry_gas_mol=dry_gas,
        temperature_k=temperature,
        total_pressure_pa=pressure,
    )

    assert y_water == pytest.approx(
        state.saturation_pressure_pa / pressure
    )


def test_subsaturated_water_remains_all_vapour() -> None:
    saturation = saturated_cathode_water_vapour_mol(
        dry_gas_mol=1.0e-3,
        temperature_k=313.15,
        total_pressure_pa=101325.0,
    )
    total_water = 0.5 * saturation

    state = repartition_cathode_water_equilibrium(
        total_water_mol=total_water,
        dry_gas_mol=1.0e-3,
        temperature_k=313.15,
        total_pressure_pa=101325.0,
    )

    assert state.vapour_mol == pytest.approx(total_water)
    assert state.liquid_mol == pytest.approx(0.0)
    assert state.relative_humidity == pytest.approx(0.5)


def test_supersaturated_water_condenses_to_saturation() -> None:
    saturation = saturated_cathode_water_vapour_mol(
        dry_gas_mol=1.0e-3,
        temperature_k=313.15,
        total_pressure_pa=101325.0,
    )
    total_water = 1.5 * saturation

    state = repartition_cathode_water_equilibrium(
        total_water_mol=total_water,
        dry_gas_mol=1.0e-3,
        temperature_k=313.15,
        total_pressure_pa=101325.0,
    )

    assert state.vapour_mol == pytest.approx(saturation)
    assert state.liquid_mol == pytest.approx(0.5 * saturation)
    assert state.relative_humidity == pytest.approx(1.0)


def test_phase_change_projection_conserves_total_water() -> None:
    initial_vapour = 2.0e-4
    initial_liquid = 3.0e-4

    transfer = equilibrium_phase_change_mol(
        initial_vapour_mol=initial_vapour,
        initial_liquid_mol=initial_liquid,
        dry_gas_mol=1.0e-3,
        temperature_k=313.15,
        total_pressure_pa=101325.0,
    )
    final_vapour = initial_vapour - transfer
    final_liquid = initial_liquid + transfer

    assert final_vapour + final_liquid == pytest.approx(
        initial_vapour + initial_liquid
    )


def test_phase_change_projection_can_evaporate_existing_liquid() -> None:
    saturation = saturated_cathode_water_vapour_mol(
        dry_gas_mol=1.0e-3,
        temperature_k=313.15,
        total_pressure_pa=101325.0,
    )
    initial_vapour = 0.5 * saturation
    initial_liquid = 0.25 * saturation

    transfer = equilibrium_phase_change_mol(
        initial_vapour_mol=initial_vapour,
        initial_liquid_mol=initial_liquid,
        dry_gas_mol=1.0e-3,
        temperature_k=313.15,
        total_pressure_pa=101325.0,
    )

    assert transfer < 0.0
