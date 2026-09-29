"""Reduced nitrogen inventory for the V0.7 dead-end anode model.

This module adds an explicit N2 inventory without yet coupling nitrogen
dilution back into electrochemistry.  The crossover source is deliberately
parametric so that conservation and purge transport can be validated before
attempting membrane-transport calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anode import (
    AnodeWaterState,
    ideal_gas_partial_pressure_pa,
    repartition_anode_water,
)


@dataclass(frozen=True)
class AnodeGasState:
    """Gas inventories for the reduced H2/N2/H2O anode control volume."""

    hydrogen_mol: float
    nitrogen_mol: float
    water: AnodeWaterState


def nitrogen_crossover_molar_rate(
    flux_mol_m2_s: float,
    active_area_m2: float,
) -> float:
    """Convert an imposed cathode-to-anode N2 flux to mol/s."""
    if flux_mol_m2_s < 0.0:
        raise ValueError("flux_mol_m2_s must be non-negative")
    if active_area_m2 <= 0.0:
        raise ValueError("active_area_m2 must be positive")
    return flux_mol_m2_s * active_area_m2






def humid_air_nitrogen_partial_pressure_pa(
    total_pressure_pa: float,
    oxygen_dry_mole_fraction: float,
    relative_humidity: float,
    saturation_water_pressure_pa: float,
) -> float:
    """Return cathode N2 partial pressure using a dry-air N2 surrogate.

    The dry non-O2 fraction is treated as N2, so argon and trace gases are
    lumped into the inert nitrogen surrogate used by this reduced model.
    """
    if total_pressure_pa <= 0.0:
        raise ValueError("total_pressure_pa must be positive")
    if not 0.0 <= oxygen_dry_mole_fraction <= 1.0:
        raise ValueError("oxygen_dry_mole_fraction must be in [0, 1]")
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be in [0, 1]")
    if saturation_water_pressure_pa < 0.0:
        raise ValueError("saturation_water_pressure_pa must be non-negative")

    water_partial_pressure = relative_humidity * saturation_water_pressure_pa
    dry_gas_pressure = max(total_pressure_pa - water_partial_pressure, 0.0)
    return (1.0 - oxygen_dry_mole_fraction) * dry_gas_pressure

def nitrogen_permeance_from_reference_flux(
    reference_flux_mol_m2_s: float,
    cathode_partial_pressure_pa: float,
) -> float:
    """Calibrate a linear N2 permeance from a zero-anode-pressure flux."""
    if reference_flux_mol_m2_s < 0.0:
        raise ValueError("reference_flux_mol_m2_s must be non-negative")
    if cathode_partial_pressure_pa <= 0.0:
        raise ValueError("cathode_partial_pressure_pa must be positive")
    return reference_flux_mol_m2_s / cathode_partial_pressure_pa


def nitrogen_pressure_driven_flux(
    permeance_mol_m2_s_pa: float,
    cathode_partial_pressure_pa: float,
    anode_partial_pressure_pa: float,
) -> float:
    """Return one-way N2 crossover flux from a linear partial-pressure law."""
    if permeance_mol_m2_s_pa < 0.0:
        raise ValueError("permeance_mol_m2_s_pa must be non-negative")
    if cathode_partial_pressure_pa < 0.0:
        raise ValueError("cathode_partial_pressure_pa must be non-negative")
    if anode_partial_pressure_pa < 0.0:
        raise ValueError("anode_partial_pressure_pa must be non-negative")
    return permeance_mol_m2_s_pa * max(
        cathode_partial_pressure_pa - anode_partial_pressure_pa,
        0.0,
    )

def total_gas_pressure_with_nitrogen_pa(
    hydrogen_mol: float,
    nitrogen_mol: float,
    water_vapor_mol: float,
    *,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Return ideal total pressure of H2 + N2 + water vapour."""
    if hydrogen_mol < 0.0 or nitrogen_mol < 0.0 or water_vapor_mol < 0.0:
        raise ValueError("gas inventories must be non-negative")
    return ideal_gas_partial_pressure_pa(
        hydrogen_mol + nitrogen_mol + water_vapor_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )


def hydrogen_moles_for_pressure_with_nitrogen(
    total_pressure_pa: float,
    *,
    nitrogen_mol: float,
    water_vapor_mol: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Return H2 inventory required to meet target pressure with N2/H2O present."""
    if total_pressure_pa <= 0.0:
        raise ValueError("total_pressure_pa must be positive")
    if nitrogen_mol < 0.0 or water_vapor_mol < 0.0:
        raise ValueError("gas inventories must be non-negative")

    non_hydrogen_pressure = ideal_gas_partial_pressure_pa(
        nitrogen_mol + water_vapor_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    hydrogen_pressure = max(total_pressure_pa - non_hydrogen_pressure, 0.0)
    return (
        hydrogen_pressure
        * volume_m3
        / (gas_constant_j_mol_k * temperature_k)
    )


def remove_well_mixed_h2_n2_h2o(
    state: AnodeGasState,
    *,
    gas_outflow_mol: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> tuple[AnodeGasState, float, float, float]:
    """Remove a finite amount from a perfectly mixed H2/N2/H2O gas phase."""
    if gas_outflow_mol < 0.0:
        raise ValueError("gas_outflow_mol must be non-negative")
    if state.hydrogen_mol < 0.0 or state.nitrogen_mol < 0.0:
        raise ValueError("gas inventories must be non-negative")

    gas_total = (
        state.hydrogen_mol
        + state.nitrogen_mol
        + state.water.vapor_mol
    )
    if gas_total <= 0.0 or gas_outflow_mol == 0.0:
        return state, 0.0, 0.0, 0.0

    removed_total = min(gas_outflow_mol, gas_total)
    h2_fraction = state.hydrogen_mol / gas_total
    n2_fraction = state.nitrogen_mol / gas_total
    water_fraction = state.water.vapor_mol / gas_total

    removed_h2 = removed_total * h2_fraction
    removed_n2 = removed_total * n2_fraction
    removed_water = removed_total * water_fraction

    remaining_water_total = (
        state.water.vapor_mol
        + state.water.liquid_mol
        - removed_water
    )
    new_water = repartition_anode_water(
        max(remaining_water_total, 0.0),
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    new_state = AnodeGasState(
        hydrogen_mol=max(state.hydrogen_mol - removed_h2, 0.0),
        nitrogen_mol=max(state.nitrogen_mol - removed_n2, 0.0),
        water=new_water,
    )
    return new_state, removed_h2, removed_n2, removed_water
