"""Reduced streamwise open-cathode airflow balances for V0.9."""

from __future__ import annotations

from dataclasses import dataclass

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
