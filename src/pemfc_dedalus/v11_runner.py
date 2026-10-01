"""Generic time-domain runner for the V11 predictive PEMFC model.

Controls are piecewise constant. Integration steps are split exactly at control
boundaries and Ballard purge events so discontinuities are never crossed by one
continuous numerical update.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .v11_purge import (
    V11PurgeClock,
    apply_v11_purge_event,
    time_to_ballard_purge_s,
)
from .v11_system import (
    V11CoupledDiagnostics,
    V11DynamicDerivative,
    V11DynamicState,
    coupled_v11_predictive_rhs,
)


@dataclass(frozen=True)
class V11ControlSegment:
    """Piecewise-constant current and cathode-air command."""

    start_time_s: float
    current_a: float
    stack_air_flow_slpm: float


@dataclass(frozen=True)
class V11RunnerInputs:
    """Time-invariant boundary conditions and electrochemical inputs."""

    inlet_air_temperature_k: float
    cathode_total_pressure_pa: float
    inlet_oxygen_mole_fraction: float
    inlet_water_mole_fraction: float
    cathode_platinum_loading_mg_cm2_geo: float
    cathode_ecsa_m2_pt_g_pt: float
    additional_resolved_loss_v: float = 0.0


@dataclass(frozen=True)
class V11TrajectoryPoint:
    """One sampled V11 dynamic state and its algebraic diagnostics."""

    time_s: float
    current_a: float
    stack_air_flow_slpm: float
    state: V11DynamicState
    diagnostics: V11CoupledDiagnostics
    purge_count: int
    charge_since_purge_as: float
    purge_event: bool


def validate_control_segments(
    segments: tuple[V11ControlSegment, ...],
) -> None:
    """Validate a piecewise-constant control schedule."""
    if not segments:
        raise ValueError("at least one control segment is required")
    if segments[0].start_time_s != 0.0:
        raise ValueError("first control segment must start at t=0")
    previous = -1.0
    for segment in segments:
        if segment.start_time_s <= previous:
            raise ValueError("control segment start times must increase")
        if segment.current_a < 0.0:
            raise ValueError("current_a must be non-negative")
        if segment.stack_air_flow_slpm <= 0.0:
            raise ValueError("stack_air_flow_slpm must be positive")
        previous = segment.start_time_s


def _control_index_at_time(
    segments: tuple[V11ControlSegment, ...],
    time_s: float,
) -> int:
    index = 0
    for candidate in range(1, len(segments)):
        if segments[candidate].start_time_s > time_s:
            break
        index = candidate
    return index


def _next_control_boundary_s(
    segments: tuple[V11ControlSegment, ...],
    index: int,
) -> float:
    if index + 1 >= len(segments):
        return float("inf")
    return segments[index + 1].start_time_s


def advance_state_euler(
    *,
    state: V11DynamicState,
    derivative: V11DynamicDerivative,
    dt_s: float,
) -> V11DynamicState:
    """Advance the V11 continuous state by one explicit Euler step."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    values: dict[str, float] = {}
    derivative_names = {
        "stack_temperature_k": "stack_temperature_k_s",
        "membrane_mean_water_content": "membrane_mean_water_content_s",
        "anode_hydrogen_mol": "anode_hydrogen_mol_s",
        "anode_nitrogen_mol": "anode_nitrogen_mol_s",
        "anode_water_vapour_mol": "anode_water_vapour_mol_s",
        "cathode_oxygen_mol": "cathode_oxygen_mol_s",
        "cathode_nitrogen_mol": "cathode_nitrogen_mol_s",
        "cathode_total_water_mol": "cathode_total_water_mol_s",
    }
    for field in fields(V11DynamicState):
        name = field.name
        rate_name = derivative_names[name]
        values[name] = getattr(state, name) + dt_s * getattr(
            derivative,
            rate_name,
        )

    updated = V11DynamicState(**values)
    if updated.stack_temperature_k <= 0.0:
        raise ValueError("integration produced non-positive stack temperature")
    if updated.membrane_mean_water_content < 0.0:
        raise ValueError("integration produced negative membrane water content")
    inventories = (
        updated.anode_hydrogen_mol,
        updated.anode_nitrogen_mol,
        updated.anode_water_vapour_mol,
        updated.cathode_oxygen_mol,
        updated.cathode_nitrogen_mol,
        updated.cathode_total_water_mol,
    )
    if min(inventories) < 0.0:
        raise ValueError(
            "integration produced a negative gas/water inventory; reduce dt_s"
        )
    return updated


def _diagnostics(
    *,
    state: V11DynamicState,
    control: V11ControlSegment,
    inputs: V11RunnerInputs,
    dt_s: float,
) -> tuple[V11DynamicDerivative, V11CoupledDiagnostics]:
    return coupled_v11_predictive_rhs(
        state=state,
        current_a=control.current_a,
        stack_air_flow_slpm=control.stack_air_flow_slpm,
        inlet_air_temperature_k=inputs.inlet_air_temperature_k,
        cathode_total_pressure_pa=inputs.cathode_total_pressure_pa,
        inlet_oxygen_mole_fraction=inputs.inlet_oxygen_mole_fraction,
        inlet_water_mole_fraction=inputs.inlet_water_mole_fraction,
        dt_regulator_s=dt_s,
        cathode_platinum_loading_mg_cm2_geo=(
            inputs.cathode_platinum_loading_mg_cm2_geo
        ),
        cathode_ecsa_m2_pt_g_pt=inputs.cathode_ecsa_m2_pt_g_pt,
        additional_resolved_loss_v=inputs.additional_resolved_loss_v,
    )


def run_v11_dynamic(
    *,
    initial_state: V11DynamicState,
    controls: tuple[V11ControlSegment, ...],
    inputs: V11RunnerInputs,
    stop_time_s: float,
    dt_s: float,
    automatic_purge: bool = True,
    manual_purge_times_s: tuple[float, ...] = (),
    initial_purge_clock: V11PurgeClock | None = None,
    sample_every_s: float | None = None,
) -> list[V11TrajectoryPoint]:
    """Integrate V11 with piecewise controls and discrete purge events.

    Explicit Euler is intentionally used as the first transparent runner. Any
    negative inventory raises instead of being clipped, making insufficient
    time resolution visible.
    """
    validate_control_segments(controls)
    if stop_time_s <= 0.0:
        raise ValueError("stop_time_s must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if sample_every_s is not None and sample_every_s <= 0.0:
        raise ValueError("sample_every_s must be positive when provided")
    if any(time <= 0.0 or time > stop_time_s for time in manual_purge_times_s):
        raise ValueError(
            "manual purge times must lie in the interval (0, stop_time_s]"
        )
    if any(
        right <= left
        for left, right in zip(manual_purge_times_s, manual_purge_times_s[1:])
    ):
        raise ValueError("manual purge times must be strictly increasing")

    state = initial_state
    clock = V11PurgeClock() if initial_purge_clock is None else initial_purge_clock
    purge_count = 0
    time_s = 0.0
    next_sample_s = 0.0
    manual_purge_index = 0
    trajectory: list[V11TrajectoryPoint] = []
    tolerance = 1.0e-12

    def append_point(*, purge_event: bool, diagnostic_dt_s: float) -> None:
        index = _control_index_at_time(controls, time_s)
        control = controls[index]
        _, diagnostics = _diagnostics(
            state=state,
            control=control,
            inputs=inputs,
            dt_s=diagnostic_dt_s,
        )
        trajectory.append(
            V11TrajectoryPoint(
                time_s=time_s,
                current_a=control.current_a,
                stack_air_flow_slpm=control.stack_air_flow_slpm,
                state=state,
                diagnostics=diagnostics,
                purge_count=purge_count,
                charge_since_purge_as=clock.charge_since_purge_as,
                purge_event=purge_event,
            )
        )

    append_point(purge_event=False, diagnostic_dt_s=dt_s)
    if sample_every_s is not None:
        next_sample_s = sample_every_s

    while time_s < stop_time_s - tolerance:
        index = _control_index_at_time(controls, time_s)
        control = controls[index]
        control_boundary = _next_control_boundary_s(controls, index)
        purge_wait = (
            time_to_ballard_purge_s(
                clock=clock,
                current_a=control.current_a,
            )
            if automatic_purge
            else float("inf")
        )
        next_manual_purge_s = (
            manual_purge_times_s[manual_purge_index]
            if manual_purge_index < len(manual_purge_times_s)
            else float("inf")
        )

        step = min(
            dt_s,
            stop_time_s - time_s,
            control_boundary - time_s,
            purge_wait,
            next_manual_purge_s - time_s,
        )
        if sample_every_s is not None:
            step = min(step, next_sample_s - time_s)

        if step <= tolerance:
            automatic_purge_now = automatic_purge and purge_wait <= tolerance
            manual_purge_now = next_manual_purge_s <= time_s + tolerance
            if automatic_purge_now or manual_purge_now:
                event = apply_v11_purge_event(state=state)
                state = event.state
                clock = V11PurgeClock()
                purge_count += 1
                if manual_purge_now:
                    manual_purge_index += 1
                append_point(purge_event=True, diagnostic_dt_s=dt_s)
                continue

            if sample_every_s is not None and next_sample_s <= time_s + tolerance:
                append_point(purge_event=False, diagnostic_dt_s=dt_s)
                next_sample_s += sample_every_s
                continue

            # A control boundary is handled by recomputing the active segment.
            time_s = min(control_boundary, stop_time_s)
            continue

        derivative, _ = _diagnostics(
            state=state,
            control=control,
            inputs=inputs,
            dt_s=step,
        )
        state = advance_state_euler(
            state=state,
            derivative=derivative,
            dt_s=step,
        )
        time_s += step
        clock = V11PurgeClock(
            charge_since_purge_as=(
                clock.charge_since_purge_as + control.current_a * step
            )
        )

        automatic_purge_now = (
            automatic_purge
            and time_to_ballard_purge_s(
                clock=clock,
                current_a=control.current_a,
            )
            <= tolerance
        )
        next_manual_purge_s = (
            manual_purge_times_s[manual_purge_index]
            if manual_purge_index < len(manual_purge_times_s)
            else float("inf")
        )
        manual_purge_now = next_manual_purge_s <= time_s + tolerance
        purge_now = automatic_purge_now or manual_purge_now
        if purge_now:
            event = apply_v11_purge_event(state=state)
            state = event.state
            clock = V11PurgeClock()
            purge_count += 1
            if manual_purge_now:
                manual_purge_index += 1
            append_point(purge_event=True, diagnostic_dt_s=dt_s)

        if sample_every_s is None:
            if not purge_now:
                append_point(purge_event=False, diagnostic_dt_s=dt_s)
        elif time_s >= next_sample_s - tolerance:
            if not purge_now:
                append_point(purge_event=False, diagnostic_dt_s=dt_s)
            next_sample_s += sample_every_s

    if trajectory[-1].time_s < stop_time_s - tolerance:
        append_point(purge_event=False, diagnostic_dt_s=dt_s)
    return trajectory



def trajectory_point_to_row(point: V11TrajectoryPoint) -> dict[str, float | int | bool]:
    """Flatten one trajectory point for CSV/JSON output."""
    diagnostics = point.diagnostics
    if diagnostics.voltage is None:
        raise ValueError("predictive trajectory requires voltage diagnostics")
    if diagnostics.nitrogen_crossover is None:
        raise ValueError("predictive trajectory requires N2 diagnostics")
    if diagnostics.heat_transfer is None:
        raise ValueError("predictive trajectory requires heat-transfer diagnostics")

    voltage = diagnostics.voltage
    crossover = diagnostics.nitrogen_crossover
    heat = diagnostics.heat_transfer
    state = point.state
    return {
        "time_s": point.time_s,
        "current_a": point.current_a,
        "stack_air_flow_slpm": point.stack_air_flow_slpm,
        "purge_event": point.purge_event,
        "purge_count": point.purge_count,
        "charge_since_purge_as": point.charge_since_purge_as,
        "stack_temperature_k": state.stack_temperature_k,
        "membrane_mean_water_content": state.membrane_mean_water_content,
        "anode_hydrogen_mol": state.anode_hydrogen_mol,
        "anode_nitrogen_mol": state.anode_nitrogen_mol,
        "anode_water_vapour_mol": state.anode_water_vapour_mol,
        "cathode_oxygen_mol": state.cathode_oxygen_mol,
        "cathode_nitrogen_mol": state.cathode_nitrogen_mol,
        "cathode_total_water_mol": state.cathode_total_water_mol,
        "anode_water_activity": diagnostics.anode_water_activity,
        "cathode_water_activity": diagnostics.cathode_water_activity,
        "cathode_water_vapour_mol": diagnostics.cathode_phase.vapour_mol,
        "cathode_liquid_water_mol": diagnostics.cathode_phase.liquid_mol,
        "cell_voltage_v": voltage.cell_voltage_v,
        "reversible_voltage_v": voltage.reversible_v,
        "orr_activation_loss_v": voltage.orr_activation_loss_v,
        "membrane_ohmic_loss_v": voltage.membrane_ohmic_loss_v,
        "additional_resolved_loss_v": voltage.additional_resolved_loss_v,
        "heat_generation_w": diagnostics.thermal.heat_generation_w,
        "air_cooling_w": diagnostics.thermal.air_cooling_w,
        "temperature_rate_k_s": diagnostics.thermal.temperature_rate_k_s,
        "air_outlet_temperature_k": heat.outlet_temperature_k,
        "heat_transfer_ntu": heat.ntu,
        "heat_transfer_effectiveness": heat.effectiveness,
        "nitrogen_crossover_flux_mol_m2_s": crossover.flux_mol_m2_s,
        "nitrogen_crossover_rate_mol_s": crossover.rate_mol_s,
        "hydrogen_inlet_mol_s": diagnostics.hydrogen_inlet_mol_s,
        "cathode_outlet_molar_flow_per_cell_mol_s": (
            diagnostics.cathode_outlet_molar_flow_per_cell_mol_s
        ),
        "water_conservation_residual_mol_s": (
            diagnostics.water_conservation_residual_mol_s
        ),
    }


def trajectory_to_rows(
    trajectory: list[V11TrajectoryPoint],
) -> list[dict[str, float | int | bool]]:
    """Flatten a complete trajectory for persistent output."""
    return [trajectory_point_to_row(point) for point in trajectory]
