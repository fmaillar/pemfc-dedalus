"""Lumped open-cathode thermal balance for V0.8."""

from __future__ import annotations


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
