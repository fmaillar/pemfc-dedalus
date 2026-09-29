"""Lumped anode gas-water inventory for V0.7.

This module introduces the first dynamic anode control-volume state while
keeping the cathode/membrane response quasi-steady.  Water transferred through
the membrane is converted from the V0.6 lambda-space flux into a molar source
term for a finite anode gas volume.

The gas phase is assumed to remain in instantaneous vapour/liquid equilibrium
at the stack temperature.  V0.7 first isolates the water-inventory dynamics, then adds a lumped
hydrogen inventory and an idealized dead-end pressure-regulated inlet before
introducing purge transients.
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



def hydrogen_consumption_molar_rate(
    current_a: float,
    faraday_c_mol: float,
) -> float:
    """Return electrochemical H2 consumption [mol/s] from I = 2 F n_dot."""
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if faraday_c_mol <= 0.0:
        raise ValueError("faraday_c_mol must be positive")
    return current_a / (2.0 * faraday_c_mol)


def ideal_gas_partial_pressure_pa(
    moles: float,
    *,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Return ideal-gas partial pressure [Pa] for one gas species."""
    if moles < 0.0:
        raise ValueError("moles must be non-negative")
    if volume_m3 <= 0.0:
        raise ValueError("volume_m3 must be positive")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")
    return moles * gas_constant_j_mol_k * temperature_k / volume_m3


def hydrogen_moles_for_total_pressure(
    total_pressure_pa: float,
    *,
    water_vapor_mol: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Return H2 moles required for a target total H2+H2O pressure."""
    if total_pressure_pa <= 0.0:
        raise ValueError("total_pressure_pa must be positive")
    if water_vapor_mol < 0.0:
        raise ValueError("water_vapor_mol must be non-negative")
    water_pressure = ideal_gas_partial_pressure_pa(
        water_vapor_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    hydrogen_pressure = max(total_pressure_pa - water_pressure, 0.0)
    return (
        hydrogen_pressure
        * volume_m3
        / (gas_constant_j_mol_k * temperature_k)
    )


def anode_total_gas_pressure_pa(
    hydrogen_mol: float,
    water_vapor_mol: float,
    *,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Return ideal total pressure [Pa] of H2 plus water vapour."""
    return ideal_gas_partial_pressure_pa(
        hydrogen_mol + water_vapor_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )


def advance_pressure_regulated_hydrogen(
    hydrogen_mol: float,
    *,
    current_a: float,
    water_vapor_mol: float,
    dt_s: float,
    target_total_pressure_pa: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
    faraday_c_mol: float,
) -> tuple[float, float, float]:
    """Advance H2 inventory with consumption and an ideal one-way regulator.

    The dead-end inlet can add hydrogen but cannot remove gas.  During each
    step, electrochemical consumption is applied first.  The regulator then
    adds the minimum H2 needed to recover the target total H2+H2O pressure.
    If water accumulation already keeps total pressure above target, the inlet
    closes and no hydrogen is added.

    Returns:
        (new_hydrogen_mol, inlet_rate_mol_s, consumption_rate_mol_s)
    """
    if hydrogen_mol < 0.0:
        raise ValueError("hydrogen_mol must be non-negative")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    consumption_rate = hydrogen_consumption_molar_rate(
        current_a,
        faraday_c_mol,
    )
    after_consumption = max(
        hydrogen_mol - consumption_rate * dt_s,
        0.0,
    )
    target_hydrogen = hydrogen_moles_for_total_pressure(
        target_total_pressure_pa,
        water_vapor_mol=water_vapor_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    inlet_mol = max(target_hydrogen - after_consumption, 0.0)
    return (
        after_consumption + inlet_mol,
        inlet_mol / dt_s,
        consumption_rate,
    )



def mixed_gas_purge_fraction(
    exchange_volume_m3: float,
    control_volume_m3: float,
) -> float:
    """Return well-mixed gas fraction removed by a purge exchange volume.

    For a perfectly mixed constant-volume control volume, displacement by an
    exchanged gas volume Vp leaves exp(-Vp / V) of the original gas inventory.
    """
    if exchange_volume_m3 < 0.0:
        raise ValueError("exchange_volume_m3 must be non-negative")
    if control_volume_m3 <= 0.0:
        raise ValueError("control_volume_m3 must be positive")
    return float(1.0 - np.exp(-exchange_volume_m3 / control_volume_m3))


def purge_anode_gas(
    hydrogen_mol: float,
    water_state: AnodeWaterState,
    *,
    purge_fraction: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> tuple[float, AnodeWaterState, float, float]:
    """Apply an instantaneous well-mixed gas purge.

    The purge removes the same fraction of H2 and gas-phase H2O.  Liquid water
    is not directly expelled, but the remaining total water inventory is
    repartitioned immediately to preserve the existing vapour/liquid
    equilibrium assumption.

    Returns:
        (new_hydrogen_mol, new_water_state, purged_hydrogen_mol,
         purged_water_mol)
    """
    if hydrogen_mol < 0.0:
        raise ValueError("hydrogen_mol must be non-negative")
    if not 0.0 <= purge_fraction <= 1.0:
        raise ValueError("purge_fraction must be in [0, 1]")

    purged_hydrogen = purge_fraction * hydrogen_mol
    purged_water = purge_fraction * water_state.vapor_mol
    remaining_hydrogen = hydrogen_mol - purged_hydrogen
    remaining_total_water = (
        water_state.vapor_mol + water_state.liquid_mol - purged_water
    )
    new_water_state = repartition_anode_water(
        max(remaining_total_water, 0.0),
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    return (
        remaining_hydrogen,
        new_water_state,
        purged_hydrogen,
        purged_water,
    )
