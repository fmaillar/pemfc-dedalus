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



def membrane_water_diffusivity_motupally(
    water_content: np.ndarray | float,
    temperature_k: float,
) -> np.ndarray:
    """Return Fickian Nafion water diffusivity from Motupally et al. [m^2/s].

    The correlation is reported in cm^2/s for Nafion 115 and is converted
    here to SI units. The two branches retain the published discontinuity
    at lambda = 3 that arises from the Darken-factor construction.

    Valid range: 0 < lambda < 17.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")

    lam = np.asarray(water_content, dtype=float)
    if np.any(lam <= 0.0) or np.any(lam >= 17.0):
        raise ValueError("water_content must satisfy 0 < lambda < 17")

    activation = np.exp(-2436.0 / temperature_k)
    low_branch_cm2_s = (
        3.10e-3
        * lam
        * (np.exp(0.28 * lam) - 1.0)
        * activation
    )
    high_branch_cm2_s = (
        4.17e-4
        * (1.0 + 161.0 * np.exp(-lam))
        * activation
    )
    diffusivity_cm2_s = np.where(
        lam <= 3.0,
        low_branch_cm2_s,
        high_branch_cm2_s,
    )
    return 1.0e-4 * diffusivity_cm2_s

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


def steady_membrane_water_profile(
    z_m: np.ndarray,
    *,
    lambda_anode: float,
    lambda_cathode: float,
    diffusivity_m2_s: float,
    drag_velocity_m_s: float,
) -> np.ndarray:
    """Return the steady 1D lambda profile for diffusion + EOD advection.

    Solves D lambda'' - v lambda' = 0 with fixed endpoint water contents.
    The small-Peclet limit is evaluated as a linear profile for numerical
    stability.
    """
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_m must be a one-dimensional grid with at least two points")
    if diffusivity_m2_s <= 0.0:
        raise ValueError("diffusivity_m2_s must be positive")
    length = float(z[-1] - z[0])
    if length <= 0.0:
        raise ValueError("z_m must be strictly increasing overall")

    xi = (z - z[0]) / length
    peclet = drag_velocity_m_s * length / diffusivity_m2_s
    if abs(peclet) < 1.0e-8:
        return lambda_anode + (lambda_cathode - lambda_anode) * xi

    denom = np.expm1(peclet)
    shape = np.expm1(peclet * xi) / denom
    return lambda_anode + (lambda_cathode - lambda_anode) * shape


def membrane_area_specific_resistance(
    z_m: np.ndarray,
    water_content: np.ndarray,
    *,
    temperature_k: float,
    conductivity_floor_s_m: float,
) -> float:
    """Return membrane protonic area-specific resistance [ohm m^2]."""
    z = np.asarray(z_m, dtype=float)
    lam = np.asarray(water_content, dtype=float)
    if z.shape != lam.shape:
        raise ValueError("z_m and water_content must have identical shapes")
    if conductivity_floor_s_m <= 0.0:
        raise ValueError("conductivity_floor_s_m must be positive")
    sigma = np.maximum(
        membrane_proton_conductivity(lam, temperature_k),
        conductivity_floor_s_m,
    )
    return float(np.trapezoid(1.0 / sigma, z))


def steady_membrane_water_profile_zero_anode_flux(
    z_m: np.ndarray,
    *,
    lambda_cathode: float,
    diffusivity_m2_s: float,
    drag_velocity_m_s: float,
) -> np.ndarray:
    """Return steady lambda with zero net water flux at the anode boundary.

    For steady 1D transport, the total lambda flux
        J = -D d(lambda)/dz + v lambda
    is spatially constant. Imposing J=0 at the anode therefore gives J=0
    throughout the membrane. With lambda fixed at the cathode boundary, the
    solution is exponential and naturally allows back-diffused water to hydrate
    the anode side even when the feed gas itself is dry.
    """
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_m must be a one-dimensional grid with at least two points")
    if diffusivity_m2_s <= 0.0:
        raise ValueError("diffusivity_m2_s must be positive")
    length = float(z[-1] - z[0])
    if length <= 0.0:
        raise ValueError("z_m must be strictly increasing overall")
    if lambda_cathode < 0.0:
        raise ValueError("lambda_cathode must be non-negative")

    return lambda_cathode * np.exp(
        drag_velocity_m_s * (z - z[-1]) / diffusivity_m2_s
    )


def steady_membrane_water_profile_anode_transfer(
    z_m: np.ndarray,
    *,
    lambda_cathode: float,
    lambda_anode_equilibrium: float,
    diffusivity_m2_s: float,
    drag_velocity_m_s: float,
    anode_transfer_coefficient_m_s: float,
) -> np.ndarray:
    """Return steady lambda with finite water transfer to the anode gas.

    The membrane coordinate increases from anode to cathode. The steady total
    lambda flux is J = -D d(lambda)/dz + v lambda, while the anode boundary
    removes water toward the gas according to
    -J = k_a (lambda_anode - lambda_anode_equilibrium).

    k_a is a phenomenological interfacial conductance in lambda-space. It is
    an explicit modelling parameter, not a Ballard-manual value. Setting
    k_a = 0 recovers the V0.5 zero-anode-flux boundary exactly.
    """
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_m must be a one-dimensional grid with at least two points")
    if diffusivity_m2_s <= 0.0:
        raise ValueError("diffusivity_m2_s must be positive")
    if anode_transfer_coefficient_m_s < 0.0:
        raise ValueError("anode_transfer_coefficient_m_s must be non-negative")
    if lambda_cathode < 0.0 or lambda_anode_equilibrium < 0.0:
        raise ValueError("membrane water contents must be non-negative")

    length = float(z[-1] - z[0])
    if length <= 0.0:
        raise ValueError("z_m must be strictly increasing overall")

    peclet = drag_velocity_m_s * length / diffusivity_m2_s
    if abs(peclet) < 1.0e-8:
        transfer_number = anode_transfer_coefficient_m_s * length / diffusivity_m2_s
        lambda_anode = (
            lambda_cathode
            + transfer_number * lambda_anode_equilibrium
        ) / (1.0 + transfer_number)
    else:
        beta = np.exp(-peclet)
        transfer_number = (
            anode_transfer_coefficient_m_s
            * (1.0 - beta)
            / drag_velocity_m_s
        )
        lambda_anode = (
            beta * lambda_cathode
            + transfer_number * lambda_anode_equilibrium
        ) / (1.0 + transfer_number)

    return steady_membrane_water_profile(
        z,
        lambda_anode=float(lambda_anode),
        lambda_cathode=lambda_cathode,
        diffusivity_m2_s=diffusivity_m2_s,
        drag_velocity_m_s=drag_velocity_m_s,
    )


def anode_water_removal_flux_lambda_m_s(
    lambda_anode: float,
    lambda_anode_equilibrium: float,
    anode_transfer_coefficient_m_s: float,
) -> float:
    """Return positive lambda-space water flux removed into the anode gas [m/s]."""
    if anode_transfer_coefficient_m_s < 0.0:
        raise ValueError("anode_transfer_coefficient_m_s must be non-negative")
    return anode_transfer_coefficient_m_s * (
        lambda_anode - lambda_anode_equilibrium
    )
