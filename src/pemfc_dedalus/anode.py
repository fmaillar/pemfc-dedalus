"""Lumped anode gas-water inventory for V0.7.

This module introduces the first dynamic anode control-volume state while
keeping the cathode/membrane response quasi-steady.  Water transferred through
the membrane is converted from the V0.6 lambda-space flux into a molar source
term for a finite anode gas volume.

The gas phase is assumed to remain in instantaneous vapour/liquid equilibrium
at the stack temperature.  Hydrogen pressure dynamics are intentionally not
included yet; V0.7 first isolates the water-inventory dynamics before adding
full dead-end H2 consumption and purge transients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnodeWaterState:
    """Water inventory in the lumped anode control volume."""

    vapor_mol: float
    liquid_mol: float
    relative_humidity: float


def water_saturation_pressure_pa(temperature_k: float) -> float:
    """Return saturation vapour pressure of water [Pa] using the Buck equation.

    The correlation is suitable for liquid water over the temperature range
    relevant to the current PEMFC model.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    temperature_c = temperature_k - 273.15
    return float(
        611.21
        * np.exp(
            (18.678 - temperature_c / 234.5)
            * (temperature_c / (257.14 + temperature_c))
        )
    )


def saturated_water_vapor_moles(
    *,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Return saturated gas-phase water inventory [mol]."""
    if volume_m3 <= 0.0:
        raise ValueError("volume_m3 must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")
    p_sat = water_saturation_pressure_pa(temperature_k)
    return p_sat * volume_m3 / (gas_constant_j_mol_k * temperature_k)


def water_vapor_moles_from_relative_humidity(
    relative_humidity: float,
    *,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Return gas-phase water moles associated with a relative humidity."""
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be in [0, 1]")
    return relative_humidity * saturated_water_vapor_moles(
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )


def lambda_flux_to_water_molar_rate(
    flux_lambda_m_s: float,
    *,
    membrane_area_m2: float,
    fixed_charge_mol_m3: float,
) -> float:
    """Convert lambda-space membrane flux to anode water source [mol/s].

    Positive V0.6 removal flux means water leaves the membrane and enters the
    anode control volume.
    """
    if membrane_area_m2 <= 0.0:
        raise ValueError("membrane_area_m2 must be positive")
    if fixed_charge_mol_m3 <= 0.0:
        raise ValueError("fixed_charge_mol_m3 must be positive")
    return flux_lambda_m_s * membrane_area_m2 * fixed_charge_mol_m3


def repartition_anode_water(
    total_water_mol: float,
    *,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> AnodeWaterState:
    """Partition total water between vapour and liquid at equilibrium."""
    if total_water_mol < 0.0:
        raise ValueError("total_water_mol must be non-negative")

    saturation_mol = saturated_water_vapor_moles(
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    vapor_mol = min(total_water_mol, saturation_mol)
    liquid_mol = max(total_water_mol - saturation_mol, 0.0)
    relative_humidity = vapor_mol / saturation_mol

    return AnodeWaterState(
        vapor_mol=vapor_mol,
        liquid_mol=liquid_mol,
        relative_humidity=relative_humidity,
    )


def advance_anode_water_state(
    state: AnodeWaterState,
    *,
    water_source_mol_s: float,
    dt_s: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> AnodeWaterState:
    """Advance the lumped anode water inventory over one explicit time step."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    total_water = state.vapor_mol + state.liquid_mol
    total_water = max(total_water + water_source_mol_s * dt_s, 0.0)
    return repartition_anode_water(
        total_water,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
