"""Solver-independent hydrated-membrane constitutive relations."""

from __future__ import annotations

from collections.abc import Callable

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

def nafion_water_interfacial_transfer_coefficient_ge(
    water_volume_fraction: np.ndarray | float,
    *,
    mode: str,
    temperature_k: float = 303.0,
    activation_energy_j_mol: float = 20_000.0,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> np.ndarray:
    """Return Ge et al. (2005) Nafion/gas water transfer coefficient [m/s].

    Reference coefficients at 303 K are:

        absorption:  k0 = 1.14e-5 m/s
        desorption:  k0 = 4.59e-5 m/s

    multiplied by membrane water volume fraction f_v and the Arrhenius factor

        exp[E_a/R * (1/303 - 1/T)].

    Ge et al. used E_a = 20 kJ/mol for the interfacial process.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if activation_energy_j_mol <= 0.0 or gas_constant_j_mol_k <= 0.0:
        raise ValueError("activation energy and gas constant must be positive")

    fraction = np.asarray(water_volume_fraction, dtype=float)
    if np.any(fraction < 0.0) or np.any(fraction > 1.0):
        raise ValueError("water_volume_fraction must be in [0, 1]")

    reference_coefficients = {
        "absorption": 1.14e-5,
        "desorption": 4.59e-5,
    }
    try:
        reference = reference_coefficients[mode]
    except KeyError as exc:
        raise ValueError("mode must be 'absorption' or 'desorption'") from exc

    arrhenius = np.exp(
        activation_energy_j_mol
        / gas_constant_j_mol_k
        * (1.0 / 303.0 - 1.0 / temperature_k)
    )
    return reference * fraction * arrhenius


def membrane_water_content_grimaldi_da(
    relative_humidity: np.ndarray | float,
    temperature_k: float,
    *,
    lambda0_eq: float = 15.01,
    adsorption_energy_j_mol: float = 1_047.0,
    exponent_eta: float = 0.4712,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> np.ndarray:
    """Return Grimaldi Dubinin-Astakhov equilibrium water content."""
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if lambda0_eq <= 0.0:
        raise ValueError("lambda0_eq must be positive")
    if adsorption_energy_j_mol <= 0.0:
        raise ValueError("adsorption_energy_j_mol must be positive")
    if exponent_eta <= 0.0:
        raise ValueError("exponent_eta must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    rh = np.asarray(relative_humidity, dtype=float)
    if np.any(rh < 0.0) or np.any(rh > 1.0):
        raise ValueError("relative_humidity must be in [0, 1]")

    result = np.zeros_like(rh, dtype=float)
    positive = rh > 0.0
    adsorption_potential = np.zeros_like(rh, dtype=float)
    adsorption_potential[positive] = (
        -gas_constant_j_mol_k
        * temperature_k
        * np.log(rh[positive])
    )
    result[positive] = lambda0_eq * np.exp(
        -(
            adsorption_potential[positive] / adsorption_energy_j_mol
        )
        ** exponent_eta
    )
    return result


def membrane_water_diffusivity_grimaldi(
    water_content: np.ndarray | float,
    temperature_k: float,
    *,
    equivalent_weight_kg_mol: float,
    dry_density_kg_m3: float,
    prefactor_m2_s: float = 6.47e-6,
    turning_point_lambda: float = 2.15,
    turning_width: float = 0.8758,
    swelling_exponent: float = -2.0,
    activation_energy_j_mol: float = 27_800.0,
    water_molar_volume_m3_mol: float = 1.8e-5,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> np.ndarray:
    """Return Grimaldi/Olesen effective Fickian water diffusivity [m^2/s]."""
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if equivalent_weight_kg_mol <= 0.0 or dry_density_kg_m3 <= 0.0:
        raise ValueError("membrane material properties must be positive")
    if prefactor_m2_s <= 0.0 or turning_width <= 0.0:
        raise ValueError("diffusivity parameters must be positive")
    if activation_energy_j_mol < 0.0:
        raise ValueError("activation_energy_j_mol must be non-negative")
    if water_molar_volume_m3_mol <= 0.0:
        raise ValueError("water_molar_volume_m3_mol must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    lam = np.asarray(water_content, dtype=float)
    if np.any(lam < 0.0):
        raise ValueError("water_content must be non-negative")

    dry_membrane_molar_volume = (
        equivalent_weight_kg_mol / dry_density_kg_m3
    )
    swelling = (
        1.0
        + water_molar_volume_m3_mol / dry_membrane_molar_volume * lam
    ) ** swelling_exponent
    hydration = 1.0 + 2.7e-3 * lam**2
    transition = 1.0 + np.tanh(
        (lam - turning_point_lambda) / turning_width
    )
    thermal = np.exp(
        -activation_energy_j_mol
        / (gas_constant_j_mol_k * temperature_k)
    )
    return prefactor_m2_s * swelling * hydration * transition * thermal


def nafion_water_interfacial_transfer_coefficient_grimaldi(
    water_content: np.ndarray | float,
    temperature_k: float,
    *,
    reference_temperature_k: float = 303.0,
    prefactor_m_s: float = 0.66e-6,
    activation_energy_j_mol: float = 6_000.0,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> np.ndarray:
    """Return Grimaldi et al. interfacial water coefficient [m/s].

    Uses the literature form

        k_g = xi_ad * lambda**1.6
              * exp[E_act / R * (1/T_ref - 1/T)].

    The default parameters reproduce the values reported for the calibrated
    PFSA interfacial transport model.
    """
    if temperature_k <= 0.0 or reference_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    if prefactor_m_s <= 0.0:
        raise ValueError("prefactor_m_s must be positive")
    if activation_energy_j_mol < 0.0:
        raise ValueError("activation_energy_j_mol must be non-negative")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    lam = np.asarray(water_content, dtype=float)
    if np.any(lam < 0.0):
        raise ValueError("water_content must be non-negative")

    temperature_factor = np.exp(
        activation_energy_j_mol
        / gas_constant_j_mol_k
        * (1.0 / reference_temperature_k - 1.0 / temperature_k)
    )
    return prefactor_m_s * lam**1.6 * temperature_factor


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



def _integrate_variable_diffusivity_flux_profile(
    z_m: np.ndarray,
    *,
    lambda_anode: float,
    flux_lambda_m_s: float,
    drag_velocity_m_s: float,
    diffusivity_model: Callable[[float], float],
    substeps_per_interval: int,
) -> np.ndarray:
    """Integrate one steady variable-D profile for a prescribed water flux."""
    if substeps_per_interval < 1:
        raise ValueError("substeps_per_interval must be >= 1")

    z = np.asarray(z_m, dtype=float)
    profile = np.empty_like(z)
    profile[0] = lambda_anode

    def derivative(value: float) -> float:
        diffusivity = float(diffusivity_model(value))
        if not np.isfinite(diffusivity) or diffusivity <= 0.0:
            raise ValueError("diffusivity_model must return finite positive values")
        return (drag_velocity_m_s * value - flux_lambda_m_s) / diffusivity

    value = float(lambda_anode)
    for index in range(z.size - 1):
        interval = float(z[index + 1] - z[index])
        if interval <= 0.0:
            raise ValueError("z_m must be strictly increasing")
        dz = interval / substeps_per_interval
        for _ in range(substeps_per_interval):
            k1 = derivative(value)
            k2 = derivative(value + 0.5 * dz * k1)
            k3 = derivative(value + 0.5 * dz * k2)
            k4 = derivative(value + dz * k3)
            value += dz * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if not 0.0 < value < 17.0 or not np.isfinite(value):
                raise ValueError("variable-diffusivity profile left physical lambda range")
        profile[index + 1] = value

    return profile


def steady_membrane_water_profile_variable_diffusivity(
    z_m: np.ndarray,
    *,
    lambda_cathode: float,
    lambda_anode_equilibrium: float,
    drag_velocity_m_s: float,
    anode_transfer_coefficient_m_s: float,
    diffusivity_model: Callable[[float], float],
    flux_scan_points: int = 129,
    root_iterations: int = 60,
    substeps_per_interval: int = 4,
) -> np.ndarray:
    """Return steady lambda profile with finite transfer and variable D.

    The unknown scalar is the through-plane water flux J_lambda.

    The Robin anode boundary gives

        lambda_anode = lambda_anode_equilibrium - J_lambda / k_a,

    while the profile satisfies

        d(lambda)/dz = (v lambda - J_lambda) / D(lambda).

    Shooting on J_lambda avoids the monotonic-profile assumption required by
    lambda-space integration and therefore remains valid when the profile has
    an interior extremum.
    """
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_m must be a one-dimensional grid with at least two points")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_m must be strictly increasing")
    if not 0.0 < lambda_cathode < 17.0:
        raise ValueError("lambda_cathode must satisfy 0 < lambda < 17")
    if not 0.0 < lambda_anode_equilibrium < 17.0:
        raise ValueError("lambda_anode_equilibrium must satisfy 0 < lambda < 17")
    if anode_transfer_coefficient_m_s <= 0.0:
        raise ValueError("anode_transfer_coefficient_m_s must be positive")
    if flux_scan_points < 3 or root_iterations < 1:
        raise ValueError("flux_scan_points and root_iterations are too small")

    transfer = anode_transfer_coefficient_m_s
    flux_lower = transfer * (lambda_anode_equilibrium - 16.999)
    flux_upper = transfer * (lambda_anode_equilibrium - 1.0e-4)

    def integrate_for_flux(flux: float) -> np.ndarray:
        lambda_anode = lambda_anode_equilibrium - flux / transfer
        return _integrate_variable_diffusivity_flux_profile(
            z,
            lambda_anode=lambda_anode,
            flux_lambda_m_s=flux,
            drag_velocity_m_s=drag_velocity_m_s,
            diffusivity_model=diffusivity_model,
            substeps_per_interval=substeps_per_interval,
        )

    def residual(flux: float) -> float:
        try:
            profile = integrate_for_flux(flux)
        except (ValueError, FloatingPointError, OverflowError):
            return float("nan")
        return float(profile[-1] - lambda_cathode)

    characteristic_flux = drag_velocity_m_s * lambda_cathode
    local_span = max(abs(characteristic_flux), 0.05 * transfer, 1.0e-10)
    candidates = [
        flux_lower,
        flux_upper,
        0.0,
        characteristic_flux,
    ]
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        candidates.extend(
            [
                characteristic_flux - factor * local_span,
                characteristic_flux + factor * local_span,
            ]
        )
    candidates.extend(
        float(value)
        for value in np.linspace(
            flux_lower,
            flux_upper,
            flux_scan_points,
        )
    )

    trials: list[tuple[float, float]] = []
    seen: set[float] = set()
    for candidate in sorted(candidates):
        flux = float(np.clip(candidate, flux_lower, flux_upper))
        key = round(flux, 16)
        if key in seen:
            continue
        seen.add(key)
        value = residual(flux)
        if np.isfinite(value):
            trials.append((flux, value))

    bracket: tuple[float, float] | None = None
    for (left, left_value), (right, right_value) in zip(
        trials,
        trials[1:],
        strict=False,
    ):
        if left_value == 0.0:
            bracket = (left, left)
            break
        if left_value * right_value <= 0.0:
            bracket = (left, right)
            break

    if bracket is None:
        if not trials:
            raise RuntimeError(
                "no physical variable-diffusivity flux-shooting trial"
            )
        best_flux, best_residual = min(
            trials,
            key=lambda item: abs(item[1]),
        )
        raise RuntimeError(
            "could not bracket variable-diffusivity flux solution; "
            f"best flux={best_flux:.6g} m/s, "
            f"lambda residual={best_residual:.6g}"
        )

    lower, upper = bracket
    if lower == upper:
        flux_solution = lower
    else:
        lower_value = residual(lower)
        for _ in range(root_iterations):
            midpoint = 0.5 * (lower + upper)
            midpoint_value = residual(midpoint)
            if not np.isfinite(midpoint_value):
                upper = midpoint
                continue
            if abs(midpoint_value) <= 1.0e-9:
                lower = midpoint
                upper = midpoint
                break
            if lower_value * midpoint_value <= 0.0:
                upper = midpoint
            else:
                lower = midpoint
                lower_value = midpoint_value
        flux_solution = 0.5 * (lower + upper)

    profile = integrate_for_flux(flux_solution)
    profile[-1] = lambda_cathode
    return profile



def steady_membrane_water_profile_variable_transfer(
    z_m: np.ndarray,
    *,
    lambda_cathode: float,
    lambda_anode_equilibrium: float,
    drag_velocity_m_s: float,
    diffusivity_model: Callable[[float], float],
    transfer_coefficient_model: Callable[[float], float],
    lambda_scan_points: int = 129,
    root_iterations: int = 60,
    substeps_per_interval: int = 4,
) -> np.ndarray:
    """Return steady lambda profile with state-dependent interfacial transfer.

    The anode boundary condition is

        J_lambda = -k(lambda_anode)
                   * (lambda_anode - lambda_anode_equilibrium).

    The unknown scalar is lambda_anode. For each trial, the corresponding
    through-plane flux is obtained from the Robin condition and the profile is
    integrated in physical z-space. This avoids assuming a monotone lambda(z).
    """
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_m must be a one-dimensional grid with at least two points")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_m must be strictly increasing")
    if not 0.0 < lambda_cathode < 17.0:
        raise ValueError("lambda_cathode must satisfy 0 < lambda < 17")
    if not 0.0 < lambda_anode_equilibrium < 17.0:
        raise ValueError("lambda_anode_equilibrium must satisfy 0 < lambda < 17")
    if lambda_scan_points < 3 or root_iterations < 1:
        raise ValueError("lambda_scan_points and root_iterations are too small")

    def flux_for(lambda_anode: float) -> float:
        coefficient = float(transfer_coefficient_model(lambda_anode))
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(
                "transfer_coefficient_model must return finite positive values"
            )
        return -coefficient * (
            lambda_anode - lambda_anode_equilibrium
        )

    def integrate_for_lambda_anode(lambda_anode: float) -> np.ndarray:
        return _integrate_variable_diffusivity_flux_profile(
            z,
            lambda_anode=lambda_anode,
            flux_lambda_m_s=flux_for(lambda_anode),
            drag_velocity_m_s=drag_velocity_m_s,
            diffusivity_model=diffusivity_model,
            substeps_per_interval=substeps_per_interval,
        )

    def residual(lambda_anode: float) -> float:
        try:
            profile = integrate_for_lambda_anode(lambda_anode)
        except (ValueError, FloatingPointError, OverflowError):
            return float("nan")
        return float(profile[-1] - lambda_cathode)

    candidates = [
        1.0e-4,
        16.999,
        lambda_anode_equilibrium,
        lambda_cathode,
        3.0 - 1.0e-6,
        3.0,
        3.0 + 1.0e-6,
    ]
    candidates.extend(
        float(value)
        for value in np.linspace(
            1.0e-4,
            16.999,
            lambda_scan_points,
        )
    )

    trials: list[tuple[float, float]] = []
    seen: set[float] = set()
    for candidate in sorted(candidates):
        value = float(np.clip(candidate, 1.0e-4, 16.999))
        key = round(value, 12)
        if key in seen:
            continue
        seen.add(key)
        residual_value = residual(value)
        if np.isfinite(residual_value):
            trials.append((value, residual_value))

    def find_bracket(
        samples: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        for (left, left_value), (right, right_value) in zip(
            samples,
            samples[1:],
            strict=False,
        ):
            if left_value == 0.0:
                return left, left
            if left_value * right_value <= 0.0:
                return left, right
        return None

    bracket = find_bracket(trials)

    if bracket is None and trials:
        best_lambda, _ = min(
            trials,
            key=lambda item: abs(item[1]),
        )
        base_spacing = 16.9989 / max(lambda_scan_points - 1, 1)

        for refinement in range(1, 9):
            half_width = base_spacing / (2.0 ** (refinement - 1))
            local_candidates = np.linspace(
                max(1.0e-4, best_lambda - half_width),
                min(16.999, best_lambda + half_width),
                33,
            )
            for candidate in local_candidates:
                value = float(candidate)
                key = round(value, 12)
                if key in seen:
                    continue
                seen.add(key)
                residual_value = residual(value)
                if np.isfinite(residual_value):
                    trials.append((value, residual_value))

            trials.sort(key=lambda item: item[0])
            bracket = find_bracket(trials)
            if bracket is not None:
                break

            best_lambda, _ = min(
                trials,
                key=lambda item: abs(item[1]),
            )

    if bracket is None:
        if not trials:
            raise RuntimeError(
                "no physical variable-transfer membrane trial"
            )
        best_lambda, best_residual = min(
            trials,
            key=lambda item: abs(item[1]),
        )
        raise RuntimeError(
            "could not bracket variable-transfer membrane solution; "
            f"best lambda_anode={best_lambda:.6g}, "
            f"lambda residual={best_residual:.6g}"
        )

    lower, upper = bracket
    if lower == upper:
        lambda_anode = lower
    else:
        lower_value = residual(lower)
        for _ in range(root_iterations):
            midpoint = 0.5 * (lower + upper)
            midpoint_value = residual(midpoint)
            if not np.isfinite(midpoint_value):
                upper = midpoint
                continue
            if abs(midpoint_value) <= 1.0e-9:
                lower = midpoint
                upper = midpoint
                break
            if lower_value * midpoint_value <= 0.0:
                upper = midpoint
            else:
                lower = midpoint
                lower_value = midpoint_value
        lambda_anode = 0.5 * (lower + upper)

    profile = integrate_for_lambda_anode(lambda_anode)
    profile[-1] = lambda_cathode
    return profile



def steady_membrane_water_profile_two_interface_transfer(
    z_m: np.ndarray,
    *,
    lambda_anode_equilibrium: float,
    lambda_cathode_equilibrium: float,
    drag_velocity_m_s: float,
    diffusivity_model: Callable[[float], float],
    transfer_coefficient_model: Callable[[float], float],
    lambda_scan_points: int = 161,
    root_iterations: int = 70,
    substeps_per_interval: int = 4,
) -> np.ndarray:
    """Return steady lambda profile with nonlinear transfer at both interfaces.

    The membrane coordinate increases from anode to cathode and

        J = -D(lambda) d(lambda)/dz + v lambda.

    At the anode:

        J = k(lambda_a) (lambda_eq,a - lambda_a)

    and at the cathode:

        J = k(lambda_c) (lambda_c - lambda_eq,c).

    The scalar shooting variable is lambda_a.
    """
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_m must be a one-dimensional grid with at least two points")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_m must be strictly increasing")
    if lambda_anode_equilibrium < 0.0:
        raise ValueError("lambda_anode_equilibrium must be non-negative")
    if lambda_cathode_equilibrium < 0.0:
        raise ValueError("lambda_cathode_equilibrium must be non-negative")
    if lambda_scan_points < 3 or root_iterations < 1:
        raise ValueError("lambda_scan_points and root_iterations are too small")

    def flux_from_anode(lambda_anode: float) -> float:
        coefficient = float(transfer_coefficient_model(lambda_anode))
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(
                "transfer_coefficient_model must return finite positive values"
            )
        return coefficient * (
            lambda_anode_equilibrium - lambda_anode
        )

    def integrate(lambda_anode: float) -> tuple[np.ndarray, float]:
        flux = flux_from_anode(lambda_anode)
        profile = _integrate_variable_diffusivity_flux_profile(
            z,
            lambda_anode=lambda_anode,
            flux_lambda_m_s=flux,
            drag_velocity_m_s=drag_velocity_m_s,
            diffusivity_model=diffusivity_model,
            substeps_per_interval=substeps_per_interval,
        )
        return profile, flux

    def residual(lambda_anode: float) -> float:
        try:
            profile, flux = integrate(lambda_anode)
            lambda_cathode = float(profile[-1])
            coefficient = float(
                transfer_coefficient_model(lambda_cathode)
            )
            if not np.isfinite(coefficient) or coefficient <= 0.0:
                return float("nan")
        except (ValueError, FloatingPointError, OverflowError):
            return float("nan")
        cathode_flux = coefficient * (
            lambda_cathode - lambda_cathode_equilibrium
        )
        return flux - cathode_flux

    candidates = [
        1.0e-4,
        16.999,
        lambda_anode_equilibrium,
        lambda_cathode_equilibrium,
        3.0 - 1.0e-6,
        3.0,
        3.0 + 1.0e-6,
    ]
    candidates.extend(
        float(value)
        for value in np.linspace(
            1.0e-4,
            16.999,
            lambda_scan_points,
        )
    )

    trials: list[tuple[float, float]] = []
    seen: set[float] = set()
    for candidate in sorted(candidates):
        value = float(np.clip(candidate, 1.0e-4, 16.999))
        key = round(value, 12)
        if key in seen:
            continue
        seen.add(key)
        residual_value = residual(value)
        if np.isfinite(residual_value):
            trials.append((value, residual_value))

    bracket: tuple[float, float] | None = None
    for (left, left_value), (right, right_value) in zip(
        trials,
        trials[1:],
        strict=False,
    ):
        if left_value == 0.0:
            bracket = (left, left)
            break
        if left_value * right_value <= 0.0:
            bracket = (left, right)
            break

    if bracket is None:
        if not trials:
            raise RuntimeError(
                "no physical two-interface transfer trial"
            )
        best_lambda, best_residual = min(
            trials,
            key=lambda item: abs(item[1]),
        )
        raise RuntimeError(
            "could not bracket two-interface transfer solution; "
            f"best lambda_anode={best_lambda:.6g}, "
            f"flux residual={best_residual:.6g} m/s"
        )

    lower, upper = bracket
    if lower == upper:
        lambda_anode = lower
    else:
        lower_value = residual(lower)
        for _ in range(root_iterations):
            midpoint = 0.5 * (lower + upper)
            midpoint_value = residual(midpoint)
            if not np.isfinite(midpoint_value):
                upper = midpoint
                continue
            if abs(midpoint_value) <= 1.0e-12:
                lower = midpoint
                upper = midpoint
                break
            if lower_value * midpoint_value <= 0.0:
                upper = midpoint
            else:
                lower = midpoint
                lower_value = midpoint_value
        lambda_anode = 0.5 * (lower + upper)

    profile, _ = integrate(lambda_anode)
    return profile


def steady_membrane_water_profile_grimaldi_consistent(
    z_m: np.ndarray,
    *,
    anode_relative_humidity: float,
    cathode_relative_humidity: float,
    temperature_k: float,
    drag_velocity_m_s: float,
    equivalent_weight_kg_mol: float,
    dry_density_kg_m3: float,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> np.ndarray:
    """Return steady Grimaldi-consistent water profile with PEMFC EOD."""
    lambda_anode_equilibrium = float(
        membrane_water_content_grimaldi_da(
            anode_relative_humidity,
            temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        ).item()
    )
    lambda_cathode_equilibrium = float(
        membrane_water_content_grimaldi_da(
            cathode_relative_humidity,
            temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        ).item()
    )

    def diffusivity_model(water_content: float) -> float:
        return float(
            membrane_water_diffusivity_grimaldi(
                water_content,
                temperature_k,
                equivalent_weight_kg_mol=equivalent_weight_kg_mol,
                dry_density_kg_m3=dry_density_kg_m3,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            ).item()
        )

    def transfer_model(water_content: float) -> float:
        return float(
            nafion_water_interfacial_transfer_coefficient_grimaldi(
                water_content,
                temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            ).item()
        )

    return steady_membrane_water_profile_two_interface_transfer(
        z_m,
        lambda_anode_equilibrium=lambda_anode_equilibrium,
        lambda_cathode_equilibrium=lambda_cathode_equilibrium,
        drag_velocity_m_s=drag_velocity_m_s,
        diffusivity_model=diffusivity_model,
        transfer_coefficient_model=transfer_model,
    )

def steady_membrane_water_profile_motupally_grimaldi(
    z_m: np.ndarray,
    *,
    lambda_cathode: float,
    lambda_anode_equilibrium: float,
    temperature_k: float,
    drag_velocity_m_s: float,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> np.ndarray:
    """Return steady Motupally profile with Grimaldi interfacial transfer."""

    def diffusivity_model(water_content: float) -> float:
        return float(
            membrane_water_diffusivity_motupally(
                water_content,
                temperature_k,
            ).item()
        )

    def transfer_model(water_content: float) -> float:
        return float(
            nafion_water_interfacial_transfer_coefficient_grimaldi(
                water_content,
                temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            ).item()
        )

    return steady_membrane_water_profile_variable_transfer(
        z_m,
        lambda_cathode=lambda_cathode,
        lambda_anode_equilibrium=lambda_anode_equilibrium,
        drag_velocity_m_s=drag_velocity_m_s,
        diffusivity_model=diffusivity_model,
        transfer_coefficient_model=transfer_model,
    )

def steady_membrane_water_profile_motupally(
    z_m: np.ndarray,
    *,
    lambda_cathode: float,
    lambda_anode_equilibrium: float,
    temperature_k: float,
    drag_velocity_m_s: float,
    anode_transfer_coefficient_m_s: float,
) -> np.ndarray:
    """Return V0.6 steady lambda profile with Motupally D(lambda, T)."""

    def diffusivity_model(water_content: float) -> float:
        return float(
            membrane_water_diffusivity_motupally(
                water_content,
                temperature_k,
            ).item()
        )

    return steady_membrane_water_profile_variable_diffusivity(
        z_m,
        lambda_cathode=lambda_cathode,
        lambda_anode_equilibrium=lambda_anode_equilibrium,
        drag_velocity_m_s=drag_velocity_m_s,
        anode_transfer_coefficient_m_s=anode_transfer_coefficient_m_s,
        diffusivity_model=diffusivity_model,
    )

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
