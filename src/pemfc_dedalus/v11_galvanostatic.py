"""Physics-first galvanostatic electrochemistry for V11.

This module contains only relations that do not require fitted PEMFC system
coefficients. Quantities that depend on stack-specific material or geometrical
properties are explicit required arguments rather than hidden defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaradayRates:
    """Per-cell electrochemical molar rates [mol/s]."""

    hydrogen_consumption_mol_s: float
    oxygen_consumption_mol_s: float
    water_production_mol_s: float


def current_density_a_m2(current_a: float, active_area_m2: float) -> float:
    """Return current density from imposed stack current and active cell area."""
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if active_area_m2 <= 0.0:
        raise ValueError("active_area_m2 must be positive")
    return current_a / active_area_m2


def faraday_rates_per_cell(
    current_a: float,
    faraday_c_mol: float = 96485.33212,
) -> FaradayRates:
    """Return exact PEMFC reaction rates implied by imposed current.

    Reaction:
        H2 + 1/2 O2 -> H2O

    Two electrons are transferred per mole of H2 and H2O, and four electrons
    per mole of O2.
    """
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if faraday_c_mol <= 0.0:
        raise ValueError("faraday_c_mol must be positive")

    hydrogen = current_a / (2.0 * faraday_c_mol)
    oxygen = current_a / (4.0 * faraday_c_mol)
    return FaradayRates(
        hydrogen_consumption_mol_s=hydrogen,
        oxygen_consumption_mol_s=oxygen,
        water_production_mol_s=hydrogen,
    )


def inverse_symmetric_butler_volmer_overpotential_v(
    *,
    current_density_a_m2: float,
    exchange_current_density_a_m2: float,
    transfer_coefficient: float,
    temperature_k: float,
    reactant_activity: float = 1.0,
    reaction_order: float = 1.0,
    faraday_c_mol: float = 96485.33212,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Return cathodic-positive activation loss from symmetric Butler-Volmer.

    The cathodic current convention is

        j = 2 j0 a^gamma sinh(F alpha eta_loss / RT),

    therefore

        eta_loss = RT/(alpha F) asinh(j / (2 j0 a^gamma)).

    No exchange-current value is supplied by default because j0 is a
    material/electrode property that must be independently sourced.
    """
    if current_density_a_m2 < 0.0:
        raise ValueError("current_density_a_m2 must be non-negative")
    if exchange_current_density_a_m2 <= 0.0:
        raise ValueError("exchange_current_density_a_m2 must be positive")
    if not 0.0 < transfer_coefficient <= 1.0:
        raise ValueError("transfer_coefficient must be in (0, 1]")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if reactant_activity <= 0.0:
        raise ValueError("reactant_activity must be positive")
    if reaction_order < 0.0:
        raise ValueError("reaction_order must be non-negative")
    if faraday_c_mol <= 0.0 or gas_constant_j_mol_k <= 0.0:
        raise ValueError("physical constants must be positive")

    beta = (
        transfer_coefficient
        * faraday_c_mol
        / (gas_constant_j_mol_k * temperature_k)
    )
    exchange = (
        exchange_current_density_a_m2
        * reactant_activity**reaction_order
    )
    return float(np.arcsinh(current_density_a_m2 / (2.0 * exchange)) / beta)


def symmetric_butler_volmer_current_density_a_m2(
    *,
    activation_loss_v: float,
    exchange_current_density_a_m2: float,
    transfer_coefficient: float,
    temperature_k: float,
    reactant_activity: float = 1.0,
    reaction_order: float = 1.0,
    faraday_c_mol: float = 96485.33212,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Return cathodic current density for the symmetric Butler-Volmer law."""
    if activation_loss_v < 0.0:
        raise ValueError("activation_loss_v must be non-negative")
    if exchange_current_density_a_m2 <= 0.0:
        raise ValueError("exchange_current_density_a_m2 must be positive")
    if not 0.0 < transfer_coefficient <= 1.0:
        raise ValueError("transfer_coefficient must be in (0, 1]")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if reactant_activity <= 0.0:
        raise ValueError("reactant_activity must be positive")
    if reaction_order < 0.0:
        raise ValueError("reaction_order must be non-negative")

    beta = (
        transfer_coefficient
        * faraday_c_mol
        / (gas_constant_j_mol_k * temperature_k)
    )
    exchange = (
        exchange_current_density_a_m2
        * reactant_activity**reaction_order
    )
    return float(2.0 * exchange * np.sinh(beta * activation_loss_v))



@dataclass(frozen=True)
class ShomateSpecies:
    """NIST Shomate thermochemistry coefficients for one temperature range."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    g: float
    h: float
    t_min_k: float
    t_max_k: float

    def sensible_enthalpy_kj_mol(self, temperature_k: float) -> float:
        """Return H(T)-H(298.15 K) from the NIST Shomate equation."""
        if not self.t_min_k <= temperature_k <= self.t_max_k:
            raise ValueError("temperature_k outside Shomate validity range")
        t = temperature_k / 1000.0
        return (
            self.a * t
            + self.b * t**2 / 2.0
            + self.c * t**3 / 3.0
            + self.d * t**4 / 4.0
            - self.e / t
            + self.f
            - self.h
        )

    def entropy_j_mol_k(self, temperature_k: float) -> float:
        """Return standard molar entropy from the NIST Shomate equation."""
        if not self.t_min_k <= temperature_k <= self.t_max_k:
            raise ValueError("temperature_k outside Shomate validity range")
        t = temperature_k / 1000.0
        return (
            self.a * np.log(t)
            + self.b * t
            + self.c * t**2 / 2.0
            + self.d * t**3 / 3.0
            - self.e / (2.0 * t**2)
            + self.g
        )

    def standard_enthalpy_kj_mol(self, temperature_k: float) -> float:
        """Return H°(T) relative to elemental reference states at 298.15 K."""
        return self.h + self.sensible_enthalpy_kj_mol(temperature_k)


_H2_SHOMATE = ShomateSpecies(
    a=33.066178,
    b=-11.363417,
    c=11.432816,
    d=-2.772874,
    e=-0.158558,
    f=-9.980797,
    g=172.707974,
    h=0.0,
    t_min_k=298.0,
    t_max_k=1000.0,
)

_O2_SHOMATE = ShomateSpecies(
    a=31.32234,
    b=-20.23531,
    c=57.86644,
    d=-36.50624,
    e=-0.007374,
    f=-8.903471,
    g=246.7945,
    h=0.0,
    t_min_k=100.0,
    t_max_k=700.0,
)

_H2O_LIQUID_SHOMATE = ShomateSpecies(
    a=-203.6060,
    b=1523.290,
    c=-3196.413,
    d=2474.455,
    e=3.855326,
    f=-256.5478,
    g=-488.7163,
    h=-285.8304,
    t_min_k=298.0,
    t_max_k=500.0,
)


def reversible_cell_voltage_liquid_water_v(
    *,
    temperature_k: float,
    hydrogen_partial_pressure_pa: float,
    oxygen_partial_pressure_pa: float,
    water_activity: float = 1.0,
    standard_pressure_pa: float = 1.0e5,
    faraday_c_mol: float = 96485.33212,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Return reversible PEMFC voltage from thermodynamics and Nernst.

    The reaction basis is H2 + 1/2 O2 -> H2O(l). Standard-state
    thermochemistry is evaluated with NIST Shomate coefficients, then corrected
    for reactant partial pressures and liquid-water activity.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if hydrogen_partial_pressure_pa <= 0.0:
        raise ValueError("hydrogen_partial_pressure_pa must be positive")
    if oxygen_partial_pressure_pa <= 0.0:
        raise ValueError("oxygen_partial_pressure_pa must be positive")
    if water_activity <= 0.0:
        raise ValueError("water_activity must be positive")
    if standard_pressure_pa <= 0.0:
        raise ValueError("standard_pressure_pa must be positive")

    delta_h_j_mol = 1000.0 * (
        _H2O_LIQUID_SHOMATE.standard_enthalpy_kj_mol(temperature_k)
        - _H2_SHOMATE.standard_enthalpy_kj_mol(temperature_k)
        - 0.5 * _O2_SHOMATE.standard_enthalpy_kj_mol(temperature_k)
    )
    delta_s_j_mol_k = (
        _H2O_LIQUID_SHOMATE.entropy_j_mol_k(temperature_k)
        - _H2_SHOMATE.entropy_j_mol_k(temperature_k)
        - 0.5 * _O2_SHOMATE.entropy_j_mol_k(temperature_k)
    )
    delta_g_j_mol = delta_h_j_mol - temperature_k * delta_s_j_mol_k
    standard_voltage = -delta_g_j_mol / (2.0 * faraday_c_mol)

    reaction_quotient_inverse = (
        (hydrogen_partial_pressure_pa / standard_pressure_pa)
        * np.sqrt(oxygen_partial_pressure_pa / standard_pressure_pa)
        / water_activity
    )
    nernst = (
        gas_constant_j_mol_k
        * temperature_k
        / (2.0 * faraday_c_mol)
        * np.log(reaction_quotient_inverse)
    )
    return float(standard_voltage + nernst)


def membrane_ohmic_loss_v(
    *,
    current_density_a_m2: float,
    membrane_thickness_m: float,
    proton_conductivity_s_m: float,
) -> float:
    """Return through-plane membrane ohmic loss j*L/sigma."""
    if current_density_a_m2 < 0.0:
        raise ValueError("current_density_a_m2 must be non-negative")
    if membrane_thickness_m <= 0.0:
        raise ValueError("membrane_thickness_m must be positive")
    if proton_conductivity_s_m <= 0.0:
        raise ValueError("proton_conductivity_s_m must be positive")
    return (
        current_density_a_m2
        * membrane_thickness_m
        / proton_conductivity_s_m
    )



@dataclass(frozen=True)
class CellVoltageBreakdown:
    """Resolved per-cell voltage terms [V]."""

    reversible_v: float
    activation_loss_v: float
    membrane_ohmic_loss_v: float
    other_resolved_loss_v: float
    cell_voltage_v: float


def resolved_cell_voltage_v(
    *,
    reversible_v: float,
    activation_loss_v: float,
    membrane_ohmic_loss_v: float,
    other_resolved_loss_v: float = 0.0,
) -> CellVoltageBreakdown:
    """Compose a cell voltage from explicitly resolved physical terms.

    This function intentionally does not invent unresolved losses. Any additional
    physical loss must be supplied explicitly.
    """
    if reversible_v <= 0.0:
        raise ValueError("reversible_v must be positive")
    for name, value in (
        ("activation_loss_v", activation_loss_v),
        ("membrane_ohmic_loss_v", membrane_ohmic_loss_v),
        ("other_resolved_loss_v", other_resolved_loss_v),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    cell_voltage = (
        reversible_v
        - activation_loss_v
        - membrane_ohmic_loss_v
        - other_resolved_loss_v
    )
    return CellVoltageBreakdown(
        reversible_v=reversible_v,
        activation_loss_v=activation_loss_v,
        membrane_ohmic_loss_v=membrane_ohmic_loss_v,
        other_resolved_loss_v=other_resolved_loss_v,
        cell_voltage_v=cell_voltage,
    )


def stack_terminal_voltage_v(
    *,
    cell_voltage_v: float,
    n_cells: int,
    current_a: float,
    bus_plate_resistance_ohm: float,
) -> float:
    """Return terminal stack voltage including the measured bus-plate loss."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if bus_plate_resistance_ohm < 0.0:
        raise ValueError("bus_plate_resistance_ohm must be non-negative")
    return (
        n_cells * cell_voltage_v
        - current_a * bus_plate_resistance_ohm
    )
