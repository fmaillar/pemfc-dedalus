"""Solver-independent hydrated-membrane constitutive relations."""

from __future__ import annotations

import numpy as np


def membrane_water_content_from_activity(
    activity: np.ndarray | float,
) -> np.ndarray:
    """Return Nafion-like membrane water content lambda from water activity.

    Uses the standard cubic Springer correlation for 0 <= a_w <= 1.
    """
    a = np.clip(np.asarray(activity, dtype=float), 0.0, 1.0)
    return 0.043 + 17.81 * a - 39.85 * a**2 + 36.0 * a**3


def electro_osmotic_drag_coefficient(
    water_content: np.ndarray | float,
) -> np.ndarray:
    """Return electro-osmotic drag coefficient n_d [mol H2O/mol H+]."""
    lam = np.maximum(np.asarray(water_content, dtype=float), 0.0)
    return lam / 22.0


def membrane_proton_conductivity(
    water_content: np.ndarray | float,
    temperature_k: float,
) -> np.ndarray:
    """Return hydrated Nafion-like proton conductivity [S/m].

    The base Springer expression is in S/cm and is converted here to S/m.
    Negative values at very low hydration are clipped to zero.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    lam = np.maximum(np.asarray(water_content, dtype=float), 0.0)
    sigma_s_cm = (0.005139 * lam - 0.00326) * np.exp(
        1268.0 * (1.0 / 303.0 - 1.0 / temperature_k)
    )
    return 100.0 * np.maximum(sigma_s_cm, 0.0)


def membrane_fixed_charge_concentration(
    dry_density_kg_m3: float,
    equivalent_weight_kg_mol: float,
) -> float:
    """Return fixed-acid-site concentration [mol/m^3]."""
    if dry_density_kg_m3 <= 0.0 or equivalent_weight_kg_mol <= 0.0:
        raise ValueError("density and equivalent weight must be positive")
    return dry_density_kg_m3 / equivalent_weight_kg_mol


def electro_osmotic_lambda_velocity(
    current_density_a_m2: float,
    faraday_c_mol: float,
    fixed_charge_mol_m3: float,
) -> float:
    """Return the lambda-advection coefficient [m/s] for n_d=lambda/22."""
    if faraday_c_mol <= 0.0 or fixed_charge_mol_m3 <= 0.0:
        raise ValueError("Faraday constant and fixed-charge concentration must be positive")
    return current_density_a_m2 / (22.0 * faraday_c_mol * fixed_charge_mol_m3)
