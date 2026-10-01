"""Periodic steady-state polarization utilities for V11.

A dead-end anode with charge-triggered purge does not have a strict steady
state. At fixed current and airflow the asymptotic regime is a purge-periodic
limit cycle. Polarization quantities are therefore evaluated over a converged
purge cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ballard_1020acs import UserStackConfiguration
from .v11_reference import reference_dynamic_scenario, reference_initial_state
from .v11_runner import (
    V11ControlSegment,
    V11RunnerInputs,
    V11TrajectoryPoint,
    run_v11_dynamic,
)
from .v11_system import V11DynamicState


@dataclass(frozen=True)
class V11PeriodicPolarizationPoint:
    """Cycle-averaged polarization result at one imposed current."""

    current_a: float
    stack_air_flow_slpm: float
    purge_period_s: float
    cycles_completed: int
    converged: bool
    cycle_state_error: float
    mean_cell_voltage_v: float
    min_cell_voltage_v: float
    max_cell_voltage_v: float
    mean_stack_temperature_k: float
    mean_membrane_water_content: float
    mean_anode_nitrogen_mol: float
    mean_stack_power_w: float
    final_state: V11DynamicState


def _state_vector(state: V11DynamicState) -> np.ndarray:
    return np.asarray(
        [
            state.stack_temperature_k,
            state.membrane_mean_water_content,
            state.anode_hydrogen_mol,
            state.anode_nitrogen_mol,
            state.anode_water_vapour_mol,
            state.cathode_oxygen_mol,
            state.cathode_nitrogen_mol,
            state.cathode_total_water_mol,
        ],
        dtype=float,
    )


def periodic_state_error(
    previous: V11DynamicState,
    current: V11DynamicState,
) -> float:
    """Return a scaled max-norm between consecutive post-purge states."""
    before = _state_vector(previous)
    after = _state_vector(current)
    floors = np.asarray(
        [300.0, 1.0, 1.0e-4, 1.0e-7, 1.0e-6, 1.0e-4, 1.0e-4, 1.0e-6],
        dtype=float,
    )
    scale = np.maximum(np.maximum(np.abs(before), np.abs(after)), floors)
    return float(np.max(np.abs(after - before) / scale))


def _cycle_points(
    trajectory: list[V11TrajectoryPoint],
    *,
    start_time_s: float,
    end_time_s: float,
) -> list[V11TrajectoryPoint]:
    return [
        point
        for point in trajectory
        if start_time_s < point.time_s <= end_time_s and not point.purge_event
    ]


def simulate_periodic_polarization_point(
    *,
    current_a: float,
    dt_s: float = 0.01,
    sample_every_s: float = 1.0,
    max_cycles: int = 12,
    convergence_tolerance: float = 1.0e-4,
    minimum_cycles: int = 3,
    initial_state: V11DynamicState | None = None,
    inputs: V11RunnerInputs | None = None,
) -> V11PeriodicPolarizationPoint:
    """Integrate one fixed-current case until successive purge cycles converge."""
    if current_a <= 0.0:
        raise ValueError("current_a must be positive")
    if dt_s <= 0.0 or sample_every_s <= 0.0:
        raise ValueError("time steps must be positive")
    if max_cycles < 2:
        raise ValueError("max_cycles must be >= 2")
    if minimum_cycles < 2 or minimum_cycles > max_cycles:
        raise ValueError("minimum_cycles must lie in [2, max_cycles]")
    if convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive")

    stack = UserStackConfiguration()
    airflow = stack.coolant_air_target_slpm(current_a)
    purge_period = stack.purge_period_s(current_a)
    scenario = reference_dynamic_scenario()
    effective_inputs = scenario.inputs if inputs is None else inputs
    state0 = reference_initial_state() if initial_state is None else initial_state

    stop_time = max_cycles * purge_period
    trajectory = run_v11_dynamic(
        initial_state=state0,
        controls=(V11ControlSegment(0.0, current_a, airflow),),
        inputs=effective_inputs,
        stop_time_s=stop_time,
        dt_s=dt_s,
        automatic_purge=True,
        sample_every_s=sample_every_s,
    )
    purge_points = [point for point in trajectory if point.purge_event]
    if len(purge_points) < 2:
        raise RuntimeError("polarization run did not produce enough purge cycles")

    converged = False
    error = float("inf")
    converged_index = len(purge_points) - 1
    for index in range(1, len(purge_points)):
        error = periodic_state_error(
            purge_points[index - 1].state,
            purge_points[index].state,
        )
        if index + 1 >= minimum_cycles and error <= convergence_tolerance:
            converged = True
            converged_index = index
            break

    cycle_end = purge_points[converged_index].time_s
    cycle_start = (
        purge_points[converged_index - 1].time_s
        if converged_index > 0
        else max(0.0, cycle_end - purge_period)
    )
    points = _cycle_points(
        trajectory,
        start_time_s=cycle_start,
        end_time_s=cycle_end,
    )
    if not points:
        raise RuntimeError("converged polarization cycle contains no samples")

    voltages = np.asarray(
        [point.diagnostics.voltage.cell_voltage_v for point in points
         if point.diagnostics.voltage is not None],
        dtype=float,
    )
    temperatures = np.asarray(
        [point.state.stack_temperature_k for point in points],
        dtype=float,
    )
    hydration = np.asarray(
        [point.state.membrane_mean_water_content for point in points],
        dtype=float,
    )
    anode_n2 = np.asarray(
        [point.state.anode_nitrogen_mol for point in points],
        dtype=float,
    )
    if voltages.size != len(points):
        raise RuntimeError("missing voltage diagnostic in polarization cycle")

    mean_voltage = float(np.mean(voltages))
    return V11PeriodicPolarizationPoint(
        current_a=current_a,
        stack_air_flow_slpm=airflow,
        purge_period_s=purge_period,
        cycles_completed=converged_index + 1,
        converged=converged,
        cycle_state_error=error,
        mean_cell_voltage_v=mean_voltage,
        min_cell_voltage_v=float(np.min(voltages)),
        max_cell_voltage_v=float(np.max(voltages)),
        mean_stack_temperature_k=float(np.mean(temperatures)),
        mean_membrane_water_content=float(np.mean(hydration)),
        mean_anode_nitrogen_mol=float(np.mean(anode_n2)),
        mean_stack_power_w=stack.stack_power_w(current_a, mean_voltage),
        final_state=purge_points[converged_index].state,
    )
