"""Tests for V11 dead-end anode and purge balances."""

from __future__ import annotations

import math

import pytest

from pemfc_dedalus.v11_anode import (
    AnodeGasState,
    anode_gas_rhs_per_cell,
    apply_well_mixed_purge,
    hydrogen_moles_for_target_pressure,
    ideal_gas_total_pressure_pa,
    mixed_purge_fraction,
    regulator_hydrogen_inlet_mol_s,
)
from pemfc_dedalus.v11_galvanostatic import faraday_rates_per_cell


def test_target_pressure_hydrogen_inventory_closes_ideal_gas_state() -> None:
    target_pressure = 101325.0 + 0.36e5
    temperature = 313.15
    volume = 10e-6
    nitrogen = 2.0e-5
    water = 1.0e-5

    hydrogen = hydrogen_moles_for_target_pressure(
        target_total_pressure_pa=target_pressure,
        nitrogen_mol=nitrogen,
        water_vapour_mol=water,
        volume_m3=volume,
        temperature_k=temperature,
    )
    state = AnodeGasState(
        hydrogen_mol=hydrogen,
        nitrogen_mol=nitrogen,
        water_vapour_mol=water,
    )

    assert ideal_gas_total_pressure_pa(
        state=state,
        volume_m3=volume,
        temperature_k=temperature,
    ) == pytest.approx(target_pressure)


def test_anode_rhs_uses_exact_faraday_hydrogen_consumption() -> None:
    current_a = 26.04
    faraday = faraday_rates_per_cell(current_a)

    rhs = anode_gas_rhs_per_cell(
        current_a=current_a,
        hydrogen_inlet_mol_s=faraday.hydrogen_consumption_mol_s,
        nitrogen_source_mol_s=0.0,
        water_source_mol_s=0.0,
    )

    assert rhs.hydrogen_mol_s == pytest.approx(0.0)
    assert rhs.nitrogen_mol_s == pytest.approx(0.0)
    assert rhs.water_vapour_mol_s == pytest.approx(0.0)


def test_ideal_regulator_replenishes_consumed_hydrogen() -> None:
    current_a = 26.04
    dt_s = 0.1
    target_pressure = 101325.0 + 0.36e5
    temperature = 313.15
    volume = 10e-6

    hydrogen = hydrogen_moles_for_target_pressure(
        target_total_pressure_pa=target_pressure,
        nitrogen_mol=0.0,
        water_vapour_mol=0.0,
        volume_m3=volume,
        temperature_k=temperature,
    )
    state = AnodeGasState(
        hydrogen_mol=hydrogen,
        nitrogen_mol=0.0,
        water_vapour_mol=0.0,
    )
    inlet = regulator_hydrogen_inlet_mol_s(
        state=state,
        target_total_pressure_pa=target_pressure,
        current_a=current_a,
        nitrogen_source_mol_s=0.0,
        water_source_mol_s=0.0,
        dt_s=dt_s,
        volume_m3=volume,
        temperature_k=temperature,
    )

    assert inlet == pytest.approx(
        faraday_rates_per_cell(current_a).hydrogen_consumption_mol_s
    )


def test_ballard_two_volume_purge_removes_one_minus_exp_minus_two() -> None:
    fraction = mixed_purge_fraction(
        purge_exchange_volume_m3=20e-6,
        anode_gas_volume_m3=10e-6,
    )

    assert fraction == pytest.approx(1.0 - math.exp(-2.0))


def test_well_mixed_purge_preserves_gas_composition() -> None:
    state = AnodeGasState(
        hydrogen_mol=8.0e-4,
        nitrogen_mol=1.5e-4,
        water_vapour_mol=0.5e-4,
    )

    result = apply_well_mixed_purge(
        state=state,
        purge_exchange_volume_m3=20e-6,
        anode_gas_volume_m3=10e-6,
    )

    before = state.mole_fractions()
    after = result.state.mole_fractions()

    assert after == pytest.approx(before)
    assert result.purge_fraction == pytest.approx(1.0 - math.exp(-2.0))
    assert (
        result.removed_hydrogen_mol
        + result.removed_nitrogen_mol
        + result.removed_water_vapour_mol
    ) == pytest.approx(result.purge_fraction * state.total_mol)
