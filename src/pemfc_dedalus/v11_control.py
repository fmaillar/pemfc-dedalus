"""Physics-based supervisory controls for the V11 digital twin.

This module deliberately avoids fitted controller gains. The steady airflow
reference is supplied by the existing periodic operating-point solver, while
thermal correction uses manufacturer temperature limits and an explicit
actuator maximum. Purge thresholds remain explicit inputs until defensible
stack-specific limits are established.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ballard_1020acs import (
    Ballard1020ACSTechnologyReference,
    UserStackConfiguration,
)
from .v11_runner import V11ControlSegment, V11PurgeCommand
from .v11_system import V11DynamicState


@dataclass(frozen=True)
class V11SupervisoryDecision:
    """One supervisory control decision."""

    stack_air_flow_slpm: float
    purge_command: bool
    target_stack_temperature_k: float
    anode_nitrogen_mole_fraction: float
    anode_water_mole_fraction: float


def _anode_mole_fractions(state: V11DynamicState) -> tuple[float, float, float]:
    total = (
        state.anode_hydrogen_mol
        + state.anode_nitrogen_mol
        + state.anode_water_vapour_mol
    )
    if total <= 0.0:
        raise ValueError("anode gas inventory must be positive")
    if min(
        state.anode_hydrogen_mol,
        state.anode_nitrogen_mol,
        state.anode_water_vapour_mol,
    ) < 0.0:
        raise ValueError("anode gas inventories must be non-negative")
    return (
        state.anode_hydrogen_mol / total,
        state.anode_nitrogen_mol / total,
        state.anode_water_vapour_mol / total,
    )


def minimum_recommended_airflow_slpm(
    current_a: float,
    *,
    stack: UserStackConfiguration | None = None,
    technology: Ballard1020ACSTechnologyReference | None = None,
) -> float:
    """Return the manufacturer-recommended minimum oxidant airflow."""
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    cfg = UserStackConfiguration() if stack is None else stack
    tech = (
        Ballard1020ACSTechnologyReference()
        if technology is None
        else technology
    )
    return cfg.cathode_air_target_slpm(
        current_a,
        stoich=tech.oxidant_stoich_min_recommended,
    )


def supervisory_control_decision(
    *,
    state: V11DynamicState,
    current_a: float,
    steady_air_flow_slpm: float,
    maximum_air_flow_slpm: float,
    max_anode_nitrogen_mole_fraction: float,
    max_anode_water_mole_fraction: float,
    stack: UserStackConfiguration | None = None,
    technology: Ballard1020ACSTechnologyReference | None = None,
) -> V11SupervisoryDecision:
    """Return a gain-free supervisory airflow and purge decision.

    steady_air_flow_slpm is the feed-forward operating-point airflow from
    the periodic V11 solver. It is floored by Ballard's recommended oxidant
    stoichiometry.

    Above the manufacturer optimum stack-temperature target, airflow is ramped
    linearly toward the explicit actuator maximum, reaching that maximum at the
    manufacturer oxidant-temperature upper limit. This is a supervisory safety
    interpolation, not a fitted PI/PID law.

    Purge thresholds are mandatory because no validated FCgen-1020ACS-specific
    nitrogen or anode-water trigger has yet been established.
    """
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if steady_air_flow_slpm < 0.0:
        raise ValueError("steady_air_flow_slpm must be non-negative")
    if maximum_air_flow_slpm <= 0.0:
        raise ValueError("maximum_air_flow_slpm must be positive")
    for name, value in (
        ("max_anode_nitrogen_mole_fraction", max_anode_nitrogen_mole_fraction),
        ("max_anode_water_mole_fraction", max_anode_water_mole_fraction),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1)")

    cfg = UserStackConfiguration() if stack is None else stack
    tech = (
        Ballard1020ACSTechnologyReference()
        if technology is None
        else technology
    )

    minimum_airflow = minimum_recommended_airflow_slpm(
        current_a,
        stack=cfg,
        technology=tech,
    )
    baseline_airflow = max(steady_air_flow_slpm, minimum_airflow)
    if maximum_air_flow_slpm < baseline_airflow:
        raise ValueError(
            "maximum_air_flow_slpm must be at least the physical baseline airflow"
        )

    target_temperature_k = (
        273.15 + tech.optimum_stack_temperature_c(current_a)
    )
    maximum_temperature_k = 273.15 + tech.oxidant_temp_max_c
    if maximum_temperature_k <= target_temperature_k:
        raise ValueError(
            "manufacturer maximum temperature must exceed target temperature"
        )

    if state.stack_temperature_k <= target_temperature_k:
        airflow = baseline_airflow
    else:
        thermal_fraction = min(
            (state.stack_temperature_k - target_temperature_k)
            / (maximum_temperature_k - target_temperature_k),
            1.0,
        )
        airflow = baseline_airflow + thermal_fraction * (
            maximum_air_flow_slpm - baseline_airflow
        )

    _, nitrogen_fraction, water_fraction = _anode_mole_fractions(state)
    purge = (
        nitrogen_fraction >= max_anode_nitrogen_mole_fraction
        or water_fraction >= max_anode_water_mole_fraction
    )

    return V11SupervisoryDecision(
        stack_air_flow_slpm=airflow,
        purge_command=purge,
        target_stack_temperature_k=target_temperature_k,
        anode_nitrogen_mole_fraction=nitrogen_fraction,
        anode_water_mole_fraction=water_fraction,
    )


def supervisory_commands_at_time(
    *,
    time_s: float,
    state: V11DynamicState,
    current_a: float,
    steady_air_flow_slpm: float,
    maximum_air_flow_slpm: float,
    max_anode_nitrogen_mole_fraction: float,
    max_anode_water_mole_fraction: float,
    stack: UserStackConfiguration | None = None,
    technology: Ballard1020ACSTechnologyReference | None = None,
) -> tuple[
    V11ControlSegment,
    tuple[V11PurgeCommand, ...],
    V11SupervisoryDecision,
]:
    """Translate one supervisory decision into V11 runner commands."""
    if time_s < 0.0:
        raise ValueError("time_s must be non-negative")

    decision = supervisory_control_decision(
        state=state,
        current_a=current_a,
        steady_air_flow_slpm=steady_air_flow_slpm,
        maximum_air_flow_slpm=maximum_air_flow_slpm,
        max_anode_nitrogen_mole_fraction=max_anode_nitrogen_mole_fraction,
        max_anode_water_mole_fraction=max_anode_water_mole_fraction,
        stack=stack,
        technology=technology,
    )
    control = V11ControlSegment(
        start_time_s=time_s,
        current_a=current_a,
        stack_air_flow_slpm=decision.stack_air_flow_slpm,
    )
    purge_commands = (
        (V11PurgeCommand(time_s=time_s),)
        if decision.purge_command and time_s > 0.0
        else ()
    )
    return control, purge_commands, decision
