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
