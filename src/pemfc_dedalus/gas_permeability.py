"""Gas-permeability conversions used by the reduced PEMFC crossover models."""

from __future__ import annotations

# 1 Barrer = 1e-10 cm^3(STP) cm / (cm^2 s cmHg).
# Converted with 22.414e-3 m^3/mol at STP and 1 cmHg = 1333.22 Pa.
BARRER_TO_MOL_M_PER_M2_S_PA = 3.348e-16


def barrer_to_si_permeability(permeability_barrer: float) -> float:
    """Convert permeability from Barrer to mol m / (m2 s Pa)."""
    if permeability_barrer < 0.0:
        raise ValueError("permeability_barrer must be non-negative")
    return permeability_barrer * BARRER_TO_MOL_M_PER_M2_S_PA


def membrane_permeance_from_permeability(
    permeability_mol_m_per_m2_s_pa: float,
    membrane_thickness_m: float,
) -> float:
    """Convert bulk membrane permeability to permeance [mol/(m2 s Pa)]."""
    if permeability_mol_m_per_m2_s_pa < 0.0:
        raise ValueError("permeability must be non-negative")
    if membrane_thickness_m <= 0.0:
        raise ValueError("membrane_thickness_m must be positive")
    return permeability_mol_m_per_m2_s_pa / membrane_thickness_m



def arrhenius_permeability(
    reference_permeability: float,
    temperature_k: float,
    reference_temperature_k: float,
    activation_energy_j_mol: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Scale permeability with an Arrhenius temperature law."""
    if reference_permeability < 0.0:
        raise ValueError("reference_permeability must be non-negative")
    if temperature_k <= 0.0 or reference_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    if activation_energy_j_mol < 0.0:
        raise ValueError("activation_energy_j_mol must be non-negative")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    exponent = (
        -activation_energy_j_mol
        / gas_constant_j_mol_k
        * (1.0 / temperature_k - 1.0 / reference_temperature_k)
    )
    return reference_permeability * float(__import__("math").exp(exponent))


def humidity_permeability_multiplier(
    relative_humidity: float,
    *,
    maximum_factor: float = 100.0,
    reference_relative_humidity: float = 0.90,
    shape_exponent: float = 2.0,
) -> float:
    """Return a bounded empirical humidity multiplier.

    Literature supports a strongly nonlinear humidity enhancement reaching up
    to roughly 100x by high RH for PFSI membranes, but does not provide one
    universal closed-form law. This interpolation is therefore explicitly a
    screening closure between dry conditions and the reported high-RH bound.
    """
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be in [0, 1]")
    if maximum_factor < 1.0:
        raise ValueError("maximum_factor must be >= 1")
    if not 0.0 < reference_relative_humidity <= 1.0:
        raise ValueError("reference_relative_humidity must be in (0, 1]")
    if shape_exponent <= 0.0:
        raise ValueError("shape_exponent must be positive")

    scaled = min(relative_humidity / reference_relative_humidity, 1.0)
    return maximum_factor ** (scaled**shape_exponent)


def effective_membrane_relative_humidity(
    anode_relative_humidity: float,
    cathode_relative_humidity: float,
) -> float:
    """Return the first-order mean boundary RH used by the screening closure."""
    if not 0.0 <= anode_relative_humidity <= 1.0:
        raise ValueError("anode_relative_humidity must be in [0, 1]")
    if not 0.0 <= cathode_relative_humidity <= 1.0:
        raise ValueError("cathode_relative_humidity must be in [0, 1]")
    return 0.5 * (anode_relative_humidity + cathode_relative_humidity)
