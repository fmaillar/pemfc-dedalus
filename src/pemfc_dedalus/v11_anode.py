"""Dead-end anode gas inventory and purge model for V11.

The control volume is one representative cell. Hydrogen consumption follows
Faraday's law exactly. Nitrogen crossover and membrane-water transfer are
explicit source terms until their transport closures are independently sourced.

The pressure-regulated H2 inlet is represented as an algebraic ideal-regulator
limit: it supplies only the hydrogen needed to recover the specified total
anode pressure and never removes gas.

A purge is represented as displacement of a perfectly mixed gas volume. This
is a reduction assumption, not a fitted coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .v11_galvanostatic import faraday_rates_per_cell


@dataclass(frozen=True)
class AnodeGasState:
    """Per-cell dead-end anode gas inventories [mol]."""

    hydrogen_mol: float
    nitrogen_mol: float
    water_vapour_mol: float

    @property
    def total_mol(self) -> float:
        return self.hydrogen_mol + self.nitrogen_mol + self.water_vapour_mol

    def mole_fractions(self) -> tuple[float, float, float]:
        """Return (y_H2, y_N2, y_H2O)."""
        if min(self.hydrogen_mol, self.nitrogen_mol, self.water_vapour_mol) < 0.0:
            raise ValueError("anode gas inventories must be non-negative")
        total = self.total_mol
        if total <= 0.0:
            raise ValueError("anode gas inventory must be positive")
        return (
            self.hydrogen_mol / total,
            self.nitrogen_mol / total,
            self.water_vapour_mol / total,
        )


@dataclass(frozen=True)
class AnodeGasDerivative:
    """Per-cell anode species accumulation rates [mol/s]."""

    hydrogen_mol_s: float
    nitrogen_mol_s: float
    water_vapour_mol_s: float


@dataclass(frozen=True)
class PurgeResult:
    """State and removed inventories after one well-mixed purge."""

    state: AnodeGasState
    removed_hydrogen_mol: float
    removed_nitrogen_mol: float
    removed_water_vapour_mol: float
    purge_fraction: float


def ideal_gas_total_pressure_pa(
    *,
    state: AnodeGasState,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Return total ideal-gas pressure for the anode control volume."""
    if volume_m3 <= 0.0:
        raise ValueError("volume_m3 must be positive")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")
    if min(state.hydrogen_mol, state.nitrogen_mol, state.water_vapour_mol) < 0.0:
        raise ValueError("anode gas inventories must be non-negative")

    return (
        state.total_mol
        * gas_constant_j_mol_k
        * temperature_k
        / volume_m3
    )


def hydrogen_moles_for_target_pressure(
    *,
    target_total_pressure_pa: float,
    nitrogen_mol: float,
    water_vapour_mol: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Return H2 inventory needed to reach a specified total pressure."""
    if target_total_pressure_pa <= 0.0:
        raise ValueError("target_total_pressure_pa must be positive")
    if nitrogen_mol < 0.0 or water_vapour_mol < 0.0:
        raise ValueError("non-hydrogen inventories must be non-negative")
    if volume_m3 <= 0.0:
        raise ValueError("volume_m3 must be positive")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")

    total_target_mol = (
        target_total_pressure_pa
        * volume_m3
        / (gas_constant_j_mol_k * temperature_k)
    )
    return max(total_target_mol - nitrogen_mol - water_vapour_mol, 0.0)


def regulator_hydrogen_inlet_mol_s(
    *,
    state: AnodeGasState,
    target_total_pressure_pa: float,
    current_a: float,
    nitrogen_source_mol_s: float,
    water_source_mol_s: float,
    dt_s: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float = 8.31446261815324,
    faraday_c_mol: float = 96485.33212,
) -> float:
    """Return ideal one-way regulator H2 flow needed over the next time step."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if nitrogen_source_mol_s < 0.0:
        raise ValueError("nitrogen_source_mol_s must be non-negative")

    faraday = faraday_rates_per_cell(current_a, faraday_c_mol)
    after_h2 = max(
        state.hydrogen_mol - faraday.hydrogen_consumption_mol_s * dt_s,
        0.0,
    )
    after_n2 = max(state.nitrogen_mol + nitrogen_source_mol_s * dt_s, 0.0)
    after_water = max(state.water_vapour_mol + water_source_mol_s * dt_s, 0.0)

    target_h2 = hydrogen_moles_for_target_pressure(
        target_total_pressure_pa=target_total_pressure_pa,
        nitrogen_mol=after_n2,
        water_vapour_mol=after_water,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    required_h2 = max(target_h2 - after_h2, 0.0)
    return required_h2 / dt_s


def anode_gas_rhs_per_cell(
    *,
    current_a: float,
    hydrogen_inlet_mol_s: float,
    nitrogen_source_mol_s: float,
    water_source_mol_s: float,
    faraday_c_mol: float = 96485.33212,
) -> AnodeGasDerivative:
    """Return dead-end anode species derivatives between purge events."""
    if hydrogen_inlet_mol_s < 0.0:
        raise ValueError("hydrogen_inlet_mol_s must be non-negative")
    if nitrogen_source_mol_s < 0.0:
        raise ValueError("nitrogen_source_mol_s must be non-negative")

    faraday = faraday_rates_per_cell(current_a, faraday_c_mol)
    return AnodeGasDerivative(
        hydrogen_mol_s=(
            hydrogen_inlet_mol_s - faraday.hydrogen_consumption_mol_s
        ),
        nitrogen_mol_s=nitrogen_source_mol_s,
        water_vapour_mol_s=water_source_mol_s,
    )


def mixed_purge_fraction(
    *,
    purge_exchange_volume_m3: float,
    anode_gas_volume_m3: float,
) -> float:
    """Return fraction of the original well-mixed gas displaced by a purge."""
    if purge_exchange_volume_m3 < 0.0:
        raise ValueError("purge_exchange_volume_m3 must be non-negative")
    if anode_gas_volume_m3 <= 0.0:
        raise ValueError("anode_gas_volume_m3 must be positive")

    return float(
        1.0 - np.exp(-purge_exchange_volume_m3 / anode_gas_volume_m3)
    )


def apply_well_mixed_purge(
    *,
    state: AnodeGasState,
    purge_exchange_volume_m3: float,
    anode_gas_volume_m3: float,
) -> PurgeResult:
    """Remove the same mixed-gas fraction from all gas species."""
    y_h2, y_n2, y_h2o = state.mole_fractions()
    fraction = mixed_purge_fraction(
        purge_exchange_volume_m3=purge_exchange_volume_m3,
        anode_gas_volume_m3=anode_gas_volume_m3,
    )
    removed_total = fraction * state.total_mol
    removed_h2 = removed_total * y_h2
    removed_n2 = removed_total * y_n2
    removed_h2o = removed_total * y_h2o

    return PurgeResult(
        state=AnodeGasState(
            hydrogen_mol=state.hydrogen_mol - removed_h2,
            nitrogen_mol=state.nitrogen_mol - removed_n2,
            water_vapour_mol=state.water_vapour_mol - removed_h2o,
        ),
        removed_hydrogen_mol=removed_h2,
        removed_nitrogen_mol=removed_n2,
        removed_water_vapour_mol=removed_h2o,
        purge_fraction=fraction,
    )
