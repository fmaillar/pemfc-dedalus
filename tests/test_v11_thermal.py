"""Tests for the V11 stack thermal balance."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_thermal import (
    ideal_equilibrated_air_temperature_rhs_k_s,
    sensible_air_cooling_w,
    stack_heat_generation_w,
    stack_temperature_rhs_k_s,
    standard_air_mass_flow_kg_s,
)


def test_ballard_heat_generation_relation() -> None:
    heat = stack_heat_generation_w(
        n_cells=10,
        current_a=26.04,
        cell_voltage_v=0.768165517,
    )

    assert heat == pytest.approx(126.2508993732)


def test_standard_air_mass_flow_uses_manual_density() -> None:
    mass_flow = standard_air_mass_flow_kg_s(
        stack_air_flow_slpm=60.0,
    )

    assert mass_flow == pytest.approx(1.293e-3)


def test_sensible_air_cooling_matches_enthalpy_rise() -> None:
    cooling = sensible_air_cooling_w(
        stack_air_flow_slpm=60.0,
        inlet_temperature_k=293.15,
        outlet_temperature_k=303.15,
    )

    assert cooling == pytest.approx(1.293e-3 * 1005.0 * 10.0)


def test_zero_net_heat_gives_zero_temperature_rate() -> None:
    n_cells = 10
    current_a = 26.04
    cell_voltage_v = 0.768165517
    inlet_temperature_k = 293.15
    stack_air_flow_slpm = 216.1

    heat = stack_heat_generation_w(
        n_cells=n_cells,
        current_a=current_a,
        cell_voltage_v=cell_voltage_v,
    )
    mass_flow = standard_air_mass_flow_kg_s(
        stack_air_flow_slpm=stack_air_flow_slpm,
    )
    outlet_temperature_k = (
        inlet_temperature_k + heat / (mass_flow * 1005.0)
    )

    balance = stack_temperature_rhs_k_s(
        n_cells=n_cells,
        current_a=current_a,
        cell_voltage_v=cell_voltage_v,
        stack_air_flow_slpm=stack_air_flow_slpm,
        inlet_temperature_k=inlet_temperature_k,
        outlet_temperature_k=outlet_temperature_k,
    )

    assert balance.net_heat_w == pytest.approx(0.0, abs=1.0e-12)
    assert balance.temperature_rate_k_s == pytest.approx(0.0, abs=1.0e-15)


def test_ideal_equilibrated_air_is_cooling_upper_bound() -> None:
    balance = ideal_equilibrated_air_temperature_rhs_k_s(
        n_cells=10,
        current_a=26.04,
        cell_voltage_v=0.768165517,
        stack_temperature_k=313.15,
        stack_air_flow_slpm=216.1,
        inlet_temperature_k=293.15,
    )

    expected_cooling = (
        standard_air_mass_flow_kg_s(stack_air_flow_slpm=216.1)
        * 1005.0
        * 20.0
    )

    assert balance.air_cooling_w == pytest.approx(expected_cooling)
    assert balance.heat_generation_w > 0.0


def test_zero_current_gives_zero_heat_generation() -> None:
    assert stack_heat_generation_w(
        n_cells=10,
        current_a=0.0,
        cell_voltage_v=1.0,
    ) == pytest.approx(0.0)
