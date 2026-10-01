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
