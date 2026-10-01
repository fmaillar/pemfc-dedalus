"""Equilibrium cathode water phase partition for V11.

Phase change is treated as a thermodynamic equilibrium constraint rather than
as an empirical first-order kinetic law. At fixed gas pressure and temperature,
the saturated water-vapour mole fraction is p_sat(T) / P.

The formulation uses the dry-gas inventory directly, so no unsourced cathode
gas volume is required.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anode import water_saturation_pressure_pa


@dataclass(frozen=True)
class CathodeWaterPhaseState:
    """Equilibrium per-cell cathode water partition [mol]."""

    vapour_mol: float
    liquid_mol: float
    saturation_vapour_mol: float
    relative_humidity: float
    saturation_pressure_pa: float


def saturated_cathode_water_vapour_mol(
    *,
    dry_gas_mol: float,
    temperature_k: float,
    total_pressure_pa: float,
) -> float:
    """Return saturated H2O vapour moles for a fixed dry-gas inventory.

    With y_w = p_sat / P and n_dry = (1-y_w) n_total,

        n_w,sat = y_w/(1-y_w) * n_dry.
    """
    if dry_gas_mol < 0.0:
        raise ValueError("dry_gas_mol must be non-negative")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if total_pressure_pa <= 0.0:
        raise ValueError("total_pressure_pa must be positive")

    saturation_pressure = water_saturation_pressure_pa(temperature_k)
    if saturation_pressure >= total_pressure_pa:
        raise ValueError(
            "water saturation pressure must be below total gas pressure"
        )

    y_sat = saturation_pressure / total_pressure_pa
    return dry_gas_mol * y_sat / (1.0 - y_sat)


def repartition_cathode_water_equilibrium(
    *,
    total_water_mol: float,
    dry_gas_mol: float,
    temperature_k: float,
    total_pressure_pa: float,
) -> CathodeWaterPhaseState:
    """Partition cathode water between vapour and liquid at equilibrium."""
    if total_water_mol < 0.0:
        raise ValueError("total_water_mol must be non-negative")

    saturation_pressure = water_saturation_pressure_pa(temperature_k)
    saturation_vapour = saturated_cathode_water_vapour_mol(
        dry_gas_mol=dry_gas_mol,
        temperature_k=temperature_k,
        total_pressure_pa=total_pressure_pa,
    )
    vapour = min(total_water_mol, saturation_vapour)
    liquid = max(total_water_mol - saturation_vapour, 0.0)

    if saturation_vapour > 0.0:
        relative_humidity = vapour / saturation_vapour
    else:
        relative_humidity = 0.0

    return CathodeWaterPhaseState(
        vapour_mol=vapour,
        liquid_mol=liquid,
        saturation_vapour_mol=saturation_vapour,
        relative_humidity=relative_humidity,
        saturation_pressure_pa=saturation_pressure,
    )


def equilibrium_phase_change_mol(
    *,
    initial_vapour_mol: float,
    initial_liquid_mol: float,
    dry_gas_mol: float,
    temperature_k: float,
    total_pressure_pa: float,
) -> float:
    """Return equilibrium vapour->liquid transfer amount [mol].

    Positive means condensation; negative means evaporation. This is an
    algebraic projection amount, not a kinetic rate.
    """
    if initial_vapour_mol < 0.0 or initial_liquid_mol < 0.0:
        raise ValueError("initial water inventories must be non-negative")

    equilibrium = repartition_cathode_water_equilibrium(
        total_water_mol=initial_vapour_mol + initial_liquid_mol,
        dry_gas_mol=dry_gas_mol,
        temperature_k=temperature_k,
        total_pressure_pa=total_pressure_pa,
    )
    return initial_vapour_mol - equilibrium.vapour_mol
