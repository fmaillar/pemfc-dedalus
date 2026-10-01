"""Globally conservative water bookkeeping for V11.

This module couples water production, gas/membrane exchange and cathode phase
partitioning without introducing a phase-change constitutive coefficient.

Sign convention:
- interface rates are positive into the membrane;
- phase-change rate is positive vapour -> liquid;
- cathode inlet water is positive into the cell;
- cathode outlet and anode purge water are positive out of the cell.

All internal transfers cancel exactly from the total-water balance.
"""

from __future__ import annotations

from dataclasses import dataclass

from .v11_galvanostatic import faraday_rates_per_cell


@dataclass(frozen=True)
class WaterInventoryState:
    """Per-cell water inventories [mol]."""

    anode_vapour_mol: float
    membrane_water_mol: float
    cathode_vapour_mol: float
    cathode_liquid_mol: float

    @property
    def total_mol(self) -> float:
        return (
            self.anode_vapour_mol
            + self.membrane_water_mol
            + self.cathode_vapour_mol
            + self.cathode_liquid_mol
        )


@dataclass(frozen=True)
class WaterInventoryDerivative:
    """Per-cell water inventory derivatives [mol/s]."""

    anode_vapour_mol_s: float
    membrane_water_mol_s: float
    cathode_vapour_mol_s: float
    cathode_liquid_mol_s: float

    @property
    def total_mol_s(self) -> float:
        return (
            self.anode_vapour_mol_s
            + self.membrane_water_mol_s
            + self.cathode_vapour_mol_s
            + self.cathode_liquid_mol_s
        )


@dataclass(frozen=True)
class WaterBalanceResidual:
    """External-source balance and residual [mol/s]."""

    electrochemical_generation_mol_s: float
    cathode_inlet_mol_s: float
    cathode_outlet_mol_s: float
    anode_purge_outlet_mol_s: float
    expected_total_rate_mol_s: float
    actual_total_rate_mol_s: float
    residual_mol_s: float


def water_inventory_rhs_per_cell(
    *,
    current_a: float,
    anode_interface_rate_into_membrane_mol_s: float,
    cathode_interface_rate_into_membrane_mol_s: float,
    cathode_phase_change_vapour_to_liquid_mol_s: float,
    cathode_inlet_water_mol_s: float,
    cathode_outlet_water_mol_s: float,
    anode_purge_water_mol_s: float,
    faraday_c_mol: float = 96485.33212,
) -> WaterInventoryDerivative:
    """Return conservative water inventory derivatives.

    Electrochemical water is generated on the cathode side and is initially
    assigned to the cathode vapour inventory. A later phase-change closure may
    transfer any excess to liquid without changing total water.
    """
    for name, value in (
        ("cathode_inlet_water_mol_s", cathode_inlet_water_mol_s),
        ("cathode_outlet_water_mol_s", cathode_outlet_water_mol_s),
        ("anode_purge_water_mol_s", anode_purge_water_mol_s),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    faraday = faraday_rates_per_cell(current_a, faraday_c_mol)
    production = faraday.water_production_mol_s

    return WaterInventoryDerivative(
        anode_vapour_mol_s=(
            -anode_interface_rate_into_membrane_mol_s
            - anode_purge_water_mol_s
        ),
        membrane_water_mol_s=(
            anode_interface_rate_into_membrane_mol_s
            + cathode_interface_rate_into_membrane_mol_s
        ),
        cathode_vapour_mol_s=(
            production
            + cathode_inlet_water_mol_s
            - cathode_outlet_water_mol_s
            - cathode_interface_rate_into_membrane_mol_s
            - cathode_phase_change_vapour_to_liquid_mol_s
        ),
        cathode_liquid_mol_s=cathode_phase_change_vapour_to_liquid_mol_s,
    )


def water_balance_residual_per_cell(
    *,
    derivative: WaterInventoryDerivative,
    current_a: float,
    cathode_inlet_water_mol_s: float,
    cathode_outlet_water_mol_s: float,
    anode_purge_water_mol_s: float,
    faraday_c_mol: float = 96485.33212,
) -> WaterBalanceResidual:
    """Return the global per-cell water conservation residual."""
    for name, value in (
        ("cathode_inlet_water_mol_s", cathode_inlet_water_mol_s),
        ("cathode_outlet_water_mol_s", cathode_outlet_water_mol_s),
        ("anode_purge_water_mol_s", anode_purge_water_mol_s),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    production = faraday_rates_per_cell(
        current_a,
        faraday_c_mol,
    ).water_production_mol_s
    expected = (
        production
        + cathode_inlet_water_mol_s
        - cathode_outlet_water_mol_s
        - anode_purge_water_mol_s
    )
    actual = derivative.total_mol_s

    return WaterBalanceResidual(
        electrochemical_generation_mol_s=production,
        cathode_inlet_mol_s=cathode_inlet_water_mol_s,
        cathode_outlet_mol_s=cathode_outlet_water_mol_s,
        anode_purge_outlet_mol_s=anode_purge_water_mol_s,
        expected_total_rate_mol_s=expected,
        actual_total_rate_mol_s=actual,
        residual_mol_s=actual - expected,
    )
