"""Lumped open-cathode thermal balance for V0.8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def slpm_to_m3_s(flow_slpm: float) -> float:
    """Convert standard litres per minute to cubic metres per second."""
    if flow_slpm < 0.0:
        raise ValueError("flow_slpm must be non-negative")
    return flow_slpm * 1.0e-3 / 60.0


def m3_s_to_slpm(flow_m3_s: float) -> float:
    """Convert cubic metres per second to standard litres per minute."""
    if flow_m3_s < 0.0:
        raise ValueError("flow_m3_s must be non-negative")
    return flow_m3_s * 60.0 * 1.0e3


def air_mass_flow_kg_s(
    flow_slpm: float,
    *,
    air_density_kg_m3: float = 1.204,
) -> float:
    """Return air mass flow from a standard volumetric flow."""
    if air_density_kg_m3 <= 0.0:
        raise ValueError("air_density_kg_m3 must be positive")
    return slpm_to_m3_s(flow_slpm) * air_density_kg_m3


def outlet_temperature_k(
    *,
    heat_rejection_w: float,
    inlet_temperature_k: float,
    air_flow_slpm: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> float:
    """Return ideal air outlet temperature from a sensible-heat balance."""
    if heat_rejection_w < 0.0:
        raise ValueError("heat_rejection_w must be non-negative")
    if inlet_temperature_k <= 0.0:
        raise ValueError("inlet_temperature_k must be positive")
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    mass_flow = air_mass_flow_kg_s(
        air_flow_slpm,
        air_density_kg_m3=air_density_kg_m3,
    )
    if mass_flow <= 0.0:
        if heat_rejection_w == 0.0:
            return inlet_temperature_k
        raise ValueError("air_flow_slpm must be positive when heat is rejected")

    return inlet_temperature_k + heat_rejection_w / (
        mass_flow * air_specific_heat_j_kg_k
    )


def required_air_flow_slpm(
    *,
    heat_rejection_w: float,
    inlet_temperature_k: float,
    maximum_outlet_temperature_k: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> float:
    """Return ideal air flow needed to absorb heat below an outlet limit."""
    if heat_rejection_w < 0.0:
        raise ValueError("heat_rejection_w must be non-negative")
    if inlet_temperature_k <= 0.0:
        raise ValueError("inlet_temperature_k must be positive")
    if maximum_outlet_temperature_k <= inlet_temperature_k:
        raise ValueError(
            "maximum_outlet_temperature_k must exceed inlet_temperature_k"
        )
    if air_density_kg_m3 <= 0.0:
        raise ValueError("air_density_kg_m3 must be positive")
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    if heat_rejection_w == 0.0:
        return 0.0

    delta_temperature = maximum_outlet_temperature_k - inlet_temperature_k
    volumetric_flow_m3_s = heat_rejection_w / (
        air_density_kg_m3
        * air_specific_heat_j_kg_k
        * delta_temperature
    )
    return m3_s_to_slpm(volumetric_flow_m3_s)


@dataclass(frozen=True)
class OpenCathodeAirflowTarget:
    """Result of the parameter-free V0.8 airflow target law."""

    status: str
    active_constraint: str
    target_air_flow_slpm: float | None
    stoichiometric_floor_slpm: float
    thermal_required_slpm: float | None


def open_cathode_airflow_target(
    *,
    heat_rejection_w: float,
    inlet_temperature_k: float,
    target_stack_temperature_k: float,
    stoichiometric_floor_slpm: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> OpenCathodeAirflowTarget:
    """Return the minimum airflow satisfying O2 and ideal thermal constraints."""
    if stoichiometric_floor_slpm < 0.0:
        raise ValueError("stoichiometric_floor_slpm must be non-negative")

    if inlet_temperature_k >= target_stack_temperature_k:
        return OpenCathodeAirflowTarget(
            status="target_unreachable",
            active_constraint="ambient_temperature",
            target_air_flow_slpm=None,
            stoichiometric_floor_slpm=stoichiometric_floor_slpm,
            thermal_required_slpm=None,
        )

    thermal_required = required_air_flow_slpm(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        maximum_outlet_temperature_k=target_stack_temperature_k,
        air_density_kg_m3=air_density_kg_m3,
        air_specific_heat_j_kg_k=air_specific_heat_j_kg_k,
    )

    if thermal_required > stoichiometric_floor_slpm:
        target = thermal_required
        active_constraint = "thermal"
    else:
        target = stoichiometric_floor_slpm
        active_constraint = "stoichiometric_floor"

    return OpenCathodeAirflowTarget(
        status="reachable",
        active_constraint=active_constraint,
        target_air_flow_slpm=target,
        stoichiometric_floor_slpm=stoichiometric_floor_slpm,
        thermal_required_slpm=thermal_required,
    )


def ideal_air_cooling_power_w(
    *,
    stack_temperature_k: float,
    inlet_temperature_k: float,
    air_flow_slpm: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> float:
    """Return ideal sensible cooling with air equilibrating to stack temperature."""
    if stack_temperature_k <= 0.0 or inlet_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    delta_temperature = max(
        stack_temperature_k - inlet_temperature_k,
        0.0,
    )
    mass_flow = air_mass_flow_kg_s(
        air_flow_slpm,
        air_density_kg_m3=air_density_kg_m3,
    )
    return mass_flow * air_specific_heat_j_kg_k * delta_temperature


def advance_lumped_stack_temperature_k(
    *,
    stack_temperature_k: float,
    inlet_temperature_k: float,
    heat_rejection_w: float,
    air_flow_slpm: float,
    thermal_mass_j_k: float,
    dt_s: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> float:
    """Advance one explicit-Euler step of the ideal lumped stack heat balance."""
    if thermal_mass_j_k <= 0.0:
        raise ValueError("thermal_mass_j_k must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if heat_rejection_w < 0.0:
        raise ValueError("heat_rejection_w must be non-negative")

    cooling_w = ideal_air_cooling_power_w(
        stack_temperature_k=stack_temperature_k,
        inlet_temperature_k=inlet_temperature_k,
        air_flow_slpm=air_flow_slpm,
        air_density_kg_m3=air_density_kg_m3,
        air_specific_heat_j_kg_k=air_specific_heat_j_kg_k,
    )
    return stack_temperature_k + (
        (heat_rejection_w - cooling_w)
        / thermal_mass_j_k
        * dt_s
    )


def lumped_thermal_time_constant_s(
    *,
    thermal_mass_j_k: float,
    air_flow_slpm: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> float:
    """Return the ideal lumped thermal time constant C_th/(m_dot c_p)."""
    if thermal_mass_j_k <= 0.0:
        raise ValueError("thermal_mass_j_k must be positive")
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    mass_flow = air_mass_flow_kg_s(
        air_flow_slpm,
        air_density_kg_m3=air_density_kg_m3,
    )
    if mass_flow <= 0.0:
        raise ValueError("air_flow_slpm must be positive")
    return thermal_mass_j_k / (
        mass_flow * air_specific_heat_j_kg_k
    )


def lumped_equilibrium_temperature_k(
    *,
    heat_rejection_w: float,
    inlet_temperature_k: float,
    air_flow_slpm: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> float:
    """Return the ideal steady stack temperature for a fixed air flow."""
    return outlet_temperature_k(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        air_flow_slpm=air_flow_slpm,
        air_density_kg_m3=air_density_kg_m3,
        air_specific_heat_j_kg_k=air_specific_heat_j_kg_k,
    )


def lumped_temperature_analytic_k(
    *,
    time_s: float,
    initial_stack_temperature_k: float,
    heat_rejection_w: float,
    inlet_temperature_k: float,
    air_flow_slpm: float,
    thermal_mass_j_k: float,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> float:
    """Return the analytic ideal lumped temperature for constant conditions."""
    if time_s < 0.0:
        raise ValueError("time_s must be non-negative")
    if initial_stack_temperature_k < inlet_temperature_k:
        raise ValueError(
            "analytic form assumes initial stack temperature >= inlet temperature"
        )

    equilibrium = lumped_equilibrium_temperature_k(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        air_flow_slpm=air_flow_slpm,
        air_density_kg_m3=air_density_kg_m3,
        air_specific_heat_j_kg_k=air_specific_heat_j_kg_k,
    )
    tau = lumped_thermal_time_constant_s(
        thermal_mass_j_k=thermal_mass_j_k,
        air_flow_slpm=air_flow_slpm,
        air_density_kg_m3=air_density_kg_m3,
        air_specific_heat_j_kg_k=air_specific_heat_j_kg_k,
    )
    return equilibrium + (
        initial_stack_temperature_k - equilibrium
    ) * np.exp(-time_s / tau)
