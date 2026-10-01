"""Tests for V11 charge-triggered purge events."""

from __future__ import annotations

import math

import pytest

from pemfc_dedalus.v11_purge import (
    V11PurgeClock,
    advance_ballard_purge_clock,
    apply_v11_purge_event,
    clock_after_purge_step,
    time_to_ballard_purge_s,
)
from pemfc_dedalus.v11_system import V11DynamicState


def _state() -> V11DynamicState:
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


def test_ballard_nominal_purge_period_is_2300_as_over_current() -> None:
    current = 26.04
    period = time_to_ballard_purge_s(
        clock=V11PurgeClock(),
        current_a=current,
    )

    assert period == pytest.approx(2300.0 / current)


def test_zero_current_stops_charge_clock() -> None:
    clock = V11PurgeClock(charge_since_purge_as=1200.0)

    assert math.isinf(time_to_ballard_purge_s(clock=clock, current_a=0.0))
    step = advance_ballard_purge_clock(
        clock=clock,
        current_a=0.0,
        dt_s=100.0,
    )
    assert not step.event_occurs
    assert step.charge_after_event_as == pytest.approx(1200.0)


def test_purge_event_is_located_inside_time_step_and_clock_restarts() -> None:
    clock = V11PurgeClock(charge_since_purge_as=2290.0)
    step = advance_ballard_purge_clock(
        clock=clock,
        current_a=20.0,
        dt_s=1.0,
    )

    assert step.event_occurs
    assert step.event_offset_s == pytest.approx(0.5)
    assert step.charge_after_event_as == pytest.approx(10.0)
    assert clock_after_purge_step(step).charge_since_purge_as == pytest.approx(
        10.0
    )


def test_default_ballard_purge_exchanges_two_anode_volumes() -> None:
    state = _state()
    event = apply_v11_purge_event(state=state)
    expected_fraction = 1.0 - math.exp(-2.0)

    assert event.exchange_volume_m3 == pytest.approx(20.0e-6)
    assert event.anode.purge_fraction == pytest.approx(expected_fraction)
    assert event.state.anode_hydrogen_mol == pytest.approx(
        state.anode_hydrogen_mol * (1.0 - expected_fraction)
    )
    assert event.state.anode_nitrogen_mol == pytest.approx(
        state.anode_nitrogen_mol * (1.0 - expected_fraction)
    )
    assert event.state.anode_water_vapour_mol == pytest.approx(
        state.anode_water_vapour_mol * (1.0 - expected_fraction)
    )


def test_purge_only_resets_anode_gas_states() -> None:
    state = _state()
    purged = apply_v11_purge_event(state=state).state

    assert purged.stack_temperature_k == state.stack_temperature_k
    assert (
        purged.membrane_mean_water_content
        == state.membrane_mean_water_content
    )
    assert purged.cathode_oxygen_mol == state.cathode_oxygen_mol
    assert purged.cathode_nitrogen_mol == state.cathode_nitrogen_mol
    assert purged.cathode_total_water_mol == state.cathode_total_water_mol


def test_manual_purge_volume_is_controllable() -> None:
    state = _state()
    one_volume = apply_v11_purge_event(
        state=state,
        purge_exchange_volume_m3=10.0e-6,
    )

    assert one_volume.anode.purge_fraction == pytest.approx(1.0 - math.exp(-1.0))
