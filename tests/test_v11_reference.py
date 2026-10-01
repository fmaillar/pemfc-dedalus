"""Tests for the V11 reference dynamic scenario."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_anode import AnodeGasState, ideal_gas_total_pressure_pa
from pemfc_dedalus.v11_heat_transfer import V11CathodeChannelGeometry
from pemfc_dedalus.v11_reference import (
    humid_air_mole_fractions,
    reference_dynamic_scenario,
    reference_initial_state,
)


def test_published_channel_geometry_implies_24_ml_per_cell() -> None:
    geometry = V11CathodeChannelGeometry()
    assert geometry.gas_volume_per_cell_m3 == pytest.approx(24.0e-6)


def test_humid_air_composition_sums_to_one() -> None:
    oxygen, nitrogen, water = humid_air_mole_fractions(
        temperature_k=293.15,
        total_pressure_pa=101325.0,
        relative_humidity=0.50,
    )

    assert oxygen + nitrogen + water == pytest.approx(1.0)
    assert water > 0.0
    assert oxygen < 0.2095


def test_reference_anode_starts_at_ballard_nominal_pressure() -> None:
    state = reference_initial_state()
    anode = AnodeGasState(
        hydrogen_mol=state.anode_hydrogen_mol,
        nitrogen_mol=state.anode_nitrogen_mol,
        water_vapour_mol=state.anode_water_vapour_mol,
    )
    pressure = ideal_gas_total_pressure_pa(
        state=anode,
        volume_m3=10.0e-6,
        temperature_k=state.stack_temperature_k,
    )

    assert pressure == pytest.approx(101325.0 + 0.36e5)


def test_reference_protocol_contains_nominal_26_a_segment() -> None:
    scenario = reference_dynamic_scenario()
    nominal = scenario.controls[2]

    assert nominal.start_time_s == pytest.approx(180.0)
    assert nominal.current_a == pytest.approx(26.04)
    assert nominal.stack_air_flow_slpm == pytest.approx(216.1)
    assert scenario.stop_time_s == pytest.approx(480.0)
