"""Tests for the generic V11 dynamic runner."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_purge import V11PurgeClock
from pemfc_dedalus.v11_runner import (
    V11ControlSegment,
    V11PurgeCommand,
    V11RunnerInputs,
    run_v11_dynamic,
    trajectory_to_rows,
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


def _inputs() -> V11RunnerInputs:
    return V11RunnerInputs(
        inlet_air_temperature_k=293.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.01,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )


def test_runner_splits_exactly_at_control_boundary() -> None:
    controls = (
        V11ControlSegment(0.0, 10.0, 120.0),
        V11ControlSegment(0.005, 20.0, 180.0),
    )
    trajectory = run_v11_dynamic(
        initial_state=_state(),
        controls=controls,
        inputs=_inputs(),
        stop_time_s=0.01,
        dt_s=0.01,
        automatic_purge=False,
    )

    boundary = [point for point in trajectory if point.time_s == pytest.approx(0.005)]
    assert boundary
    assert boundary[-1].current_a == pytest.approx(20.0)
    assert boundary[-1].stack_air_flow_slpm == pytest.approx(180.0)


def test_runner_applies_automatic_purge_at_exact_charge_threshold() -> None:
    trajectory = run_v11_dynamic(
        initial_state=_state(),
        controls=(V11ControlSegment(0.0, 20.0, 180.0),),
        inputs=_inputs(),
        stop_time_s=0.01,
        dt_s=0.01,
        automatic_purge=True,
        initial_purge_clock=V11PurgeClock(charge_since_purge_as=2299.9),
    )

    purge_points = [point for point in trajectory if point.purge_event]
    assert len(purge_points) == 1
    assert purge_points[0].time_s == pytest.approx(0.005)
    assert purge_points[0].purge_count == 1
    assert purge_points[0].charge_since_purge_as == pytest.approx(0.0)


def test_runner_supports_explicit_purge_command_and_resets_charge_clock() -> None:
    trajectory = run_v11_dynamic(
        initial_state=_state(),
        controls=(V11ControlSegment(0.0, 10.0, 120.0),),
        inputs=_inputs(),
        stop_time_s=0.01,
        dt_s=0.01,
        automatic_purge=False,
        purge_commands=(V11PurgeCommand(time_s=0.004),),
    )

    purge_points = [point for point in trajectory if point.purge_event]
    assert len(purge_points) == 1
    assert purge_points[0].time_s == pytest.approx(0.004)
    assert purge_points[0].charge_since_purge_as == pytest.approx(0.0)


def test_runner_serialization_exposes_dynamic_voltage_and_states() -> None:
    trajectory = run_v11_dynamic(
        initial_state=_state(),
        controls=(V11ControlSegment(0.0, 10.0, 120.0),),
        inputs=_inputs(),
        stop_time_s=0.002,
        dt_s=0.001,
        automatic_purge=False,
    )
    rows = trajectory_to_rows(trajectory)

    assert rows
    assert "cell_voltage_v" in rows[0]
    assert "stack_temperature_k" in rows[0]
    assert "membrane_mean_water_content" in rows[0]
    assert "nitrogen_crossover_rate_mol_s" in rows[0]
    assert "air_outlet_temperature_k" in rows[0]
    assert "purge_event" in rows[0]



def test_sample_at_control_boundary_uses_new_command() -> None:
    trajectory = run_v11_dynamic(
        initial_state=_state(),
        controls=(
            V11ControlSegment(0.0, 5.0, 100.0),
            V11ControlSegment(0.06, 15.0, 150.0),
        ),
        inputs=_inputs(),
        stop_time_s=0.08,
        dt_s=0.01,
        automatic_purge=False,
        sample_every_s=0.02,
    )

    boundary = [
        point
        for point in trajectory
        if point.time_s == pytest.approx(0.06)
    ]
    assert len(boundary) == 1
    assert boundary[0].current_a == pytest.approx(15.0)
    assert boundary[0].stack_air_flow_slpm == pytest.approx(150.0)


def test_sample_times_are_snapped_to_requested_grid() -> None:
    trajectory = run_v11_dynamic(
        initial_state=_state(),
        controls=(V11ControlSegment(0.0, 5.0, 100.0),),
        inputs=_inputs(),
        stop_time_s=0.06,
        dt_s=0.01,
        automatic_purge=False,
        sample_every_s=0.02,
    )

    assert [point.time_s for point in trajectory] == pytest.approx(
        [0.0, 0.02, 0.04, 0.06]
    )


def test_runner_rejects_two_purge_command_interfaces_at_once() -> None:
    with pytest.raises(ValueError, match="not both"):
        run_v11_dynamic(
            initial_state=_state(),
            controls=(V11ControlSegment(0.0, 10.0, 120.0),),
            inputs=_inputs(),
            stop_time_s=0.01,
            dt_s=0.01,
            automatic_purge=False,
            purge_commands=(V11PurgeCommand(time_s=0.004),),
            manual_purge_times_s=(0.006,),
        )


def test_v11_state_space_interface_classifies_inputs() -> None:
    from pemfc_dedalus.v11_interfaces import (
        V11ExogenousInputs,
        V11ManipulatedInputs,
        V11ModelParameters,
    )

    manipulated = V11ManipulatedInputs(216.1, False)
    exogenous = V11ExogenousInputs(
        26.04, 293.15, 101325.0, 0.21, 0.01
    )
    parameters = V11ModelParameters(0.4, 50.0)

    assert manipulated.stack_air_flow_slpm == 216.1
    assert exogenous.current_a == 26.04
    assert parameters.membrane_conductivity_multiplier == 1.0
