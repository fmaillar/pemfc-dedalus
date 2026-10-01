"""Reduced streamwise open-cathode airflow balances for V0.9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np

STANDARD_MOLAR_VOLUME_M3_MOL = 22.414e-3


def molar_flow_from_slpm(
    flow_slpm: float,
    *,
    standard_molar_volume_m3_mol: float = STANDARD_MOLAR_VOLUME_M3_MOL,
) -> float:
    """Convert standard litres per minute to molar flow."""
    if flow_slpm < 0.0:
        raise ValueError("flow_slpm must be non-negative")
    if standard_molar_volume_m3_mol <= 0.0:
        raise ValueError("standard_molar_volume_m3_mol must be positive")
    return flow_slpm * 1.0e-3 / 60.0 / standard_molar_volume_m3_mol


def oxygen_consumption_mol_s(
    current_a: float,
    *,
    faraday_c_mol: float,
) -> float:
    """Return cathode O2 consumption for one PEMFC cell."""
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if faraday_c_mol <= 0.0:
        raise ValueError("faraday_c_mol must be positive")
    return current_a / (4.0 * faraday_c_mol)


@dataclass(frozen=True)
class StreamwiseOxygenProfile:
    """One-dimensional conservative plug-flow oxygen profile."""

    streamwise_fraction: np.ndarray
    oxygen_mole_fraction: np.ndarray
    oxygen_molar_flow_mol_s: np.ndarray
    total_molar_flow_mol_s: np.ndarray
    inlet_air_flow_slpm_per_cell: float
    oxygen_stoichiometry: float
    oxygen_utilization: float

    @property
    def inlet_oxygen_mole_fraction(self) -> float:
        return float(self.oxygen_mole_fraction[0])

    @property
    def outlet_oxygen_mole_fraction(self) -> float:
        return float(self.oxygen_mole_fraction[-1])

    @property
    def relative_oxygen_mole_fraction_drop(self) -> float:
        inlet = self.inlet_oxygen_mole_fraction
        if inlet == 0.0:
            return 0.0
        return (inlet - self.outlet_oxygen_mole_fraction) / inlet


def streamwise_oxygen_profile(
    *,
    current_a: float,
    total_air_flow_slpm: float,
    n_cells: int,
    oxygen_mole_fraction: float,
    faraday_c_mol: float,
    points: int = 101,
    standard_molar_volume_m3_mol: float = STANDARD_MOLAR_VOLUME_M3_MOL,
) -> StreamwiseOxygenProfile:
    """Return a reduced O2 profile for equally split stack airflow."""
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if not 0.0 < oxygen_mole_fraction < 1.0:
        raise ValueError("oxygen_mole_fraction must lie in (0, 1)")
    if points < 2:
        raise ValueError("points must be >= 2")

    per_cell_air_flow_slpm = total_air_flow_slpm / n_cells
    inlet_total_molar_flow = molar_flow_from_slpm(
        per_cell_air_flow_slpm,
        standard_molar_volume_m3_mol=standard_molar_volume_m3_mol,
    )
    inlet_oxygen_flow = oxygen_mole_fraction * inlet_total_molar_flow
    inlet_inert_flow = (1.0 - oxygen_mole_fraction) * inlet_total_molar_flow
    consumed_oxygen = oxygen_consumption_mol_s(
        current_a,
        faraday_c_mol=faraday_c_mol,
    )

    if consumed_oxygen > inlet_oxygen_flow:
        raise ValueError("airflow cannot supply the requested oxygen consumption")

    coordinate = np.linspace(0.0, 1.0, points)
    oxygen_flow = inlet_oxygen_flow - consumed_oxygen * coordinate
    total_flow = oxygen_flow + inlet_inert_flow
    local_oxygen_fraction = oxygen_flow / total_flow

    if consumed_oxygen == 0.0:
        stoichiometry = float("inf")
        utilization = 0.0
    else:
        stoichiometry = inlet_oxygen_flow / consumed_oxygen
        utilization = consumed_oxygen / inlet_oxygen_flow

    return StreamwiseOxygenProfile(
        streamwise_fraction=coordinate,
        oxygen_mole_fraction=local_oxygen_fraction,
        oxygen_molar_flow_mol_s=oxygen_flow,
        total_molar_flow_mol_s=total_flow,
        inlet_air_flow_slpm_per_cell=per_cell_air_flow_slpm,
        oxygen_stoichiometry=stoichiometry,
        oxygen_utilization=utilization,
    )


@dataclass(frozen=True)
class StreamwiseThermalProfile:
    """One-dimensional sensible-heating profile for cathode air."""

    streamwise_fraction: np.ndarray
    air_temperature_k: np.ndarray
    inlet_air_flow_slpm_per_cell: float
    heat_rejection_w_per_cell: float

    @property
    def inlet_temperature_k(self) -> float:
        return float(self.air_temperature_k[0])

    @property
    def outlet_temperature_k(self) -> float:
        return float(self.air_temperature_k[-1])

    @property
    def temperature_rise_k(self) -> float:
        return self.outlet_temperature_k - self.inlet_temperature_k


def streamwise_air_temperature_profile(
    *,
    inlet_temperature_k: float,
    total_air_flow_slpm: float,
    n_cells: int,
    total_heat_rejection_w: float,
    points: int = 101,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> StreamwiseThermalProfile:
    """Return the plug-flow sensible-heating profile of cathode air."""
    if inlet_temperature_k <= 0.0:
        raise ValueError("inlet_temperature_k must be positive")
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if total_heat_rejection_w < 0.0:
        raise ValueError("total_heat_rejection_w must be non-negative")
    if points < 2:
        raise ValueError("points must be >= 2")
    if air_density_kg_m3 <= 0.0:
        raise ValueError("air_density_kg_m3 must be positive")
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    per_cell_air_flow_slpm = total_air_flow_slpm / n_cells
    volumetric_flow_m3_s = per_cell_air_flow_slpm * 1.0e-3 / 60.0
    mass_flow_kg_s = volumetric_flow_m3_s * air_density_kg_m3
    heat_rejection_w_per_cell = total_heat_rejection_w / n_cells

    if mass_flow_kg_s <= 0.0:
        if heat_rejection_w_per_cell == 0.0:
            coordinate = np.linspace(0.0, 1.0, points)
            temperature = np.full(points, inlet_temperature_k)
            return StreamwiseThermalProfile(
                streamwise_fraction=coordinate,
                air_temperature_k=temperature,
                inlet_air_flow_slpm_per_cell=per_cell_air_flow_slpm,
                heat_rejection_w_per_cell=heat_rejection_w_per_cell,
            )
        raise ValueError("airflow must be positive when heat is rejected")

    total_rise_k = heat_rejection_w_per_cell / (
        mass_flow_kg_s * air_specific_heat_j_kg_k
    )
    coordinate = np.linspace(0.0, 1.0, points)
    temperature = inlet_temperature_k + total_rise_k * coordinate

    return StreamwiseThermalProfile(
        streamwise_fraction=coordinate,
        air_temperature_k=temperature,
        inlet_air_flow_slpm_per_cell=per_cell_air_flow_slpm,
        heat_rejection_w_per_cell=heat_rejection_w_per_cell,
    )


@overload
def ideal_gas_species_concentration_mol_m3(
    *,
    mole_fraction: np.ndarray,
    pressure_pa: float,
    temperature_k: float | np.ndarray,
    gas_constant_j_mol_k: float,
) -> np.ndarray: ...


@overload
def ideal_gas_species_concentration_mol_m3(
    *,
    mole_fraction: float,
    pressure_pa: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
) -> float: ...


def ideal_gas_species_concentration_mol_m3(
    *,
    mole_fraction: float | np.ndarray,
    pressure_pa: float,
    temperature_k: float | np.ndarray,
    gas_constant_j_mol_k: float,
) -> float | np.ndarray:
    """Return ideal-gas species concentration y p / (R T)."""
    if pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    mole_fraction_array = np.asarray(mole_fraction, dtype=float)
    temperature_array = np.asarray(temperature_k, dtype=float)

    if np.any((mole_fraction_array < 0.0) | (mole_fraction_array > 1.0)):
        raise ValueError("mole_fraction must lie in [0, 1]")
    if np.any(temperature_array <= 0.0):
        raise ValueError("temperature_k must be positive")

    concentration = (
        mole_fraction_array
        * pressure_pa
        / (gas_constant_j_mol_k * temperature_array)
    )
    if concentration.ndim == 0:
        return float(concentration)
    return concentration
