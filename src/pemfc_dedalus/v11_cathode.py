"""Reduced cathode gas inventory balances for V11.

The control volume is one representative cell. Stack air flow is divided by the
number of parallel cell cathode passages. Species balances are strictly molar
and use Faraday's law for oxygen consumption.

Water transfer into the cathode gas phase is an explicit input. This prevents
the gas model from silently assuming that all electrochemically produced water
immediately becomes vapour.
"""

from __future__ import annotations

from dataclasses import dataclass

from .v11_galvanostatic import faraday_rates_per_cell


@dataclass(frozen=True)
class CathodeGasState:
    """Per-cell cathode gas inventories [mol]."""

    oxygen_mol: float
    nitrogen_mol: float
    water_vapour_mol: float

    @property
    def total_mol(self) -> float:
        return self.oxygen_mol + self.nitrogen_mol + self.water_vapour_mol

    def mole_fractions(self) -> tuple[float, float, float]:
        """Return (y_O2, y_N2, y_H2O)."""
        total = self.total_mol
        if total <= 0.0:
            raise ValueError("cathode gas inventory must be positive")
        if min(self.oxygen_mol, self.nitrogen_mol, self.water_vapour_mol) < 0.0:
            raise ValueError("cathode gas inventories must be non-negative")
        return (
            self.oxygen_mol / total,
            self.nitrogen_mol / total,
            self.water_vapour_mol / total,
        )


@dataclass(frozen=True)
class CathodeGasDerivative:
    """Per-cell species accumulation rates [mol/s]."""

    oxygen_mol_s: float
    nitrogen_mol_s: float
    water_vapour_mol_s: float

    @property
    def total_mol_s(self) -> float:
        return self.oxygen_mol_s + self.nitrogen_mol_s + self.water_vapour_mol_s


def standard_litre_per_minute_to_mol_s(
    flow_slpm: float,
    *,
    standard_temperature_k: float = 273.15,
    standard_pressure_pa: float = 101325.0,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Convert standard L/min to mol/s using the ideal-gas law."""
    if flow_slpm < 0.0:
        raise ValueError("flow_slpm must be non-negative")
    if standard_temperature_k <= 0.0:
        raise ValueError("standard_temperature_k must be positive")
    if standard_pressure_pa <= 0.0:
        raise ValueError("standard_pressure_pa must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    volume_flow_m3_s = flow_slpm * 1.0e-3 / 60.0
    return (
        standard_pressure_pa
        * volume_flow_m3_s
        / (gas_constant_j_mol_k * standard_temperature_k)
    )


def cathode_constant_inventory_outlet_mol_s(
    *,
    inlet_air_mol_s: float,
    current_a: float,
    water_source_to_gas_mol_s: float,
    faraday_c_mol: float = 96485.33212,
) -> float:
    """Outlet flow that keeps total cathode gas moles constant.

    With O2 electrochemical consumption and an explicit water-vapour source,

        dn_tot/dt = n_in - n_out - n_O2,rxn + n_H2O,gas.

    Setting dn_tot/dt = 0 gives the outlet closure below.
    """
    if inlet_air_mol_s < 0.0:
        raise ValueError("inlet_air_mol_s must be non-negative")
    rates = faraday_rates_per_cell(current_a, faraday_c_mol)
    outlet = (
        inlet_air_mol_s
        - rates.oxygen_consumption_mol_s
        + water_source_to_gas_mol_s
    )
    if outlet < 0.0:
        raise ValueError("constant-inventory outlet flow would be negative")
    return outlet


def cathode_gas_rhs_per_cell(
    *,
    state: CathodeGasState,
    stack_air_flow_slpm: float,
    n_cells: int,
    current_a: float,
    inlet_oxygen_mole_fraction: float,
    inlet_water_mole_fraction: float,
    water_source_to_gas_mol_s: float,
    outlet_molar_flow_per_cell_mol_s: float,
    faraday_c_mol: float = 96485.33212,
) -> CathodeGasDerivative:
    """Return one-cell cathode gas species derivatives.

    Nitrogen is the remainder of the inlet mixture. The outlet is assumed
    perfectly mixed with the control-volume composition.
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if stack_air_flow_slpm < 0.0:
        raise ValueError("stack_air_flow_slpm must be non-negative")
    if outlet_molar_flow_per_cell_mol_s < 0.0:
        raise ValueError("outlet_molar_flow_per_cell_mol_s must be non-negative")
    if not 0.0 <= inlet_oxygen_mole_fraction <= 1.0:
        raise ValueError("inlet_oxygen_mole_fraction must be in [0, 1]")
    if not 0.0 <= inlet_water_mole_fraction <= 1.0:
        raise ValueError("inlet_water_mole_fraction must be in [0, 1]")

    inlet_nitrogen_mole_fraction = (
        1.0 - inlet_oxygen_mole_fraction - inlet_water_mole_fraction
    )
    if inlet_nitrogen_mole_fraction < 0.0:
        raise ValueError("inlet gas mole fractions must sum to at most one")

    inlet_per_cell_mol_s = standard_litre_per_minute_to_mol_s(
        stack_air_flow_slpm / n_cells
    )
    y_o2, y_n2, y_h2o = state.mole_fractions()
    faraday = faraday_rates_per_cell(current_a, faraday_c_mol)

    return CathodeGasDerivative(
        oxygen_mol_s=(
            inlet_per_cell_mol_s * inlet_oxygen_mole_fraction
            - outlet_molar_flow_per_cell_mol_s * y_o2
            - faraday.oxygen_consumption_mol_s
        ),
        nitrogen_mol_s=(
            inlet_per_cell_mol_s * inlet_nitrogen_mole_fraction
            - outlet_molar_flow_per_cell_mol_s * y_n2
        ),
        water_vapour_mol_s=(
            inlet_per_cell_mol_s * inlet_water_mole_fraction
            - outlet_molar_flow_per_cell_mol_s * y_h2o
            + water_source_to_gas_mol_s
        ),
    )
