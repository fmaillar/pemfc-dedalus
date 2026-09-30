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



def _variable_diffusivity_transport_length(
    *,
    lambda_anode: float,
    lambda_cathode: float,
    flux_lambda_m_s: float,
    drag_velocity_m_s: float,
    diffusivity_model: Callable[[float], float],
    quadrature_points: int,
) -> float:
    """Return membrane length implied by one steady monotone lambda path."""
    if quadrature_points < 17:
        raise ValueError("quadrature_points must be >= 17")
    if lambda_anode == lambda_cathode:
        return 0.0

    lambda_path = np.linspace(
        lambda_anode,
        lambda_cathode,
        quadrature_points,
    )
    diffusivity = np.asarray(
        [diffusivity_model(float(value)) for value in lambda_path],
        dtype=float,
    )
    if np.any(~np.isfinite(diffusivity)) or np.any(diffusivity <= 0.0):
        raise ValueError("diffusivity_model must return finite positive values")

    denominator = drag_velocity_m_s * lambda_path - flux_lambda_m_s
    scale = max(
        float(np.max(np.abs(drag_velocity_m_s * lambda_path))),
        abs(flux_lambda_m_s),
        1.0e-30,
    )
    if np.any(np.abs(denominator) <= 1.0e-12 * scale):
        raise ValueError("steady lambda path contains a transport singularity")

    dz_dlambda = diffusivity / denominator
    increments = (
        0.5
        * (dz_dlambda[:-1] + dz_dlambda[1:])
        * np.diff(lambda_path)
    )
    if np.any(increments <= 0.0):
        raise ValueError("steady lambda path is not monotone in z")
    return float(np.sum(increments))


def _reconstruct_variable_diffusivity_profile(
    z_m: np.ndarray,
    *,
    lambda_anode: float,
    lambda_cathode: float,
    flux_lambda_m_s: float,
    drag_velocity_m_s: float,
    diffusivity_model: Callable[[float], float],
    quadrature_points: int,
) -> np.ndarray:
    """Reconstruct lambda(z) once the anode boundary value is known."""
    z = np.asarray(z_m, dtype=float)
    if lambda_anode == lambda_cathode:
        return np.full_like(z, lambda_anode)

    lambda_path = np.linspace(
        lambda_anode,
        lambda_cathode,
        quadrature_points,
    )
    diffusivity = np.asarray(
        [diffusivity_model(float(value)) for value in lambda_path],
        dtype=float,
    )
    denominator = drag_velocity_m_s * lambda_path - flux_lambda_m_s
    dz_dlambda = diffusivity / denominator
    increments = (
        0.5
        * (dz_dlambda[:-1] + dz_dlambda[1:])
        * np.diff(lambda_path)
    )
    cumulative_z = np.concatenate(([0.0], np.cumsum(increments)))
    return np.interp(z - z[0], cumulative_z, lambda_path)


def steady_membrane_water_profile_variable_diffusivity(
    z_m: np.ndarray,
    *,
    lambda_cathode: float,
    lambda_anode_equilibrium: float,
    drag_velocity_m_s: float,
    anode_transfer_coefficient_m_s: float,
    diffusivity_model: Callable[[float], float],
    lambda_lower_bound: float = 1.0e-6,
    lambda_upper_bound: float = 16.999,
    scan_points: int = 65,
    root_iterations: int = 60,
    quadrature_points: int = 129,
    initial_lambda_anode: float | None = None,
) -> np.ndarray:
    """Return steady lambda with finite transfer and variable diffusivity.

    The autonomous steady flux equation is integrated in lambda-space:

        dz/dlambda = D(lambda) / (v lambda - J),

    with

        J = -k_a (lambda_anode - lambda_anode_equilibrium).

    This avoids unstable forward shooting through very-low-diffusivity states.
    Invalid or singular trial paths are skipped while bracketing the scalar
    anode boundary value.
    """
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_m must be a one-dimensional grid with at least two points")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_m must be strictly increasing")
    if lambda_cathode < 0.0 or lambda_anode_equilibrium < 0.0:
        raise ValueError("membrane water contents must be non-negative")
    if anode_transfer_coefficient_m_s < 0.0:
        raise ValueError("anode_transfer_coefficient_m_s must be non-negative")
    if not 0.0 < lambda_lower_bound < lambda_upper_bound:
        raise ValueError("invalid lambda bounds")
    if scan_points < 3 or root_iterations < 1:
        raise ValueError("scan_points and root_iterations are too small")

    length = float(z[-1] - z[0])

    def flux_for(lambda_anode: float) -> float:
        if anode_transfer_coefficient_m_s == 0.0:
            return 0.0
        return -anode_transfer_coefficient_m_s * (
            lambda_anode - lambda_anode_equilibrium
        )

    def residual(lambda_anode: float) -> float:
        try:
            transport_length = _variable_diffusivity_transport_length(
                lambda_anode=lambda_anode,
                lambda_cathode=lambda_cathode,
                flux_lambda_m_s=flux_for(lambda_anode),
                drag_velocity_m_s=drag_velocity_m_s,
                diffusivity_model=diffusivity_model,
                quadrature_points=quadrature_points,
            )
        except (ValueError, FloatingPointError, OverflowError):
            return float("nan")
        return transport_length - length

    def evaluate_candidates(values: list[float]) -> list[tuple[float, float]]:
        trials: list[tuple[float, float]] = []
        seen: set[float] = set()
        for candidate in sorted(values):
            value = float(np.clip(
                candidate,
                lambda_lower_bound,
                lambda_upper_bound,
            ))
            key = round(value, 12)
            if key in seen:
                continue
            seen.add(key)
            residual_value = residual(value)
            if np.isfinite(residual_value):
                trials.append((value, residual_value))
        return trials

    center = (
        float(initial_lambda_anode)
        if initial_lambda_anode is not None
        else 0.5 * (lambda_anode_equilibrium + lambda_cathode)
    )
    span = max(
        abs(lambda_cathode - lambda_anode_equilibrium),
        0.25,
    )
    local_candidates = [
        lambda_lower_bound,
        lambda_upper_bound,
        lambda_anode_equilibrium,
        lambda_cathode,
        3.0 - 1.0e-6,
        3.0,
        3.0 + 1.0e-6,
        center,
    ]
    for factor in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
        local_candidates.extend(
            [center - factor * span, center + factor * span]
        )

    finite_trials = evaluate_candidates(local_candidates)

    def find_bracket(
        trials: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        for (left, left_residual), (right, right_residual) in zip(
            trials,
            trials[1:],
            strict=False,
        ):
            if left_residual == 0.0:
                return (left, left)
            if left_residual * right_residual <= 0.0:
                return (left, right)
        return None

    bracket = find_bracket(finite_trials)

    if bracket is None:
        global_candidates = list(
            np.linspace(
                lambda_lower_bound,
                lambda_upper_bound,
                scan_points,
            )
        )
        finite_trials = evaluate_candidates(
            local_candidates + global_candidates
        )
        bracket = find_bracket(finite_trials)

    if bracket is None:
        if not finite_trials:
            raise RuntimeError(
                "no physical variable-diffusivity membrane trial path"
            )
        best_value, best_residual = min(
            finite_trials,
            key=lambda item: abs(item[1]),
        )
        raise RuntimeError(
            "could not bracket variable-diffusivity membrane solution; "
            f"best lambda_anode={best_value:.6g}, "
            f"length residual={best_residual:.6g} m"
        )

    lower, upper = bracket
    if lower == upper:
        lambda_anode = lower
    else:
        lower_residual = residual(lower)
        for _ in range(root_iterations):
            midpoint = 0.5 * (lower + upper)
            midpoint_residual = residual(midpoint)
            if not np.isfinite(midpoint_residual):
                upper = midpoint
                continue
            if abs(midpoint_residual) <= 1.0e-10 * max(length, 1.0e-12):
                lower = midpoint
                upper = midpoint
                break
            if lower_residual * midpoint_residual <= 0.0:
                upper = midpoint
            else:
                lower = midpoint
                lower_residual = midpoint_residual
        lambda_anode = 0.5 * (lower + upper)

    flux_lambda_m_s = flux_for(lambda_anode)
    profile = _reconstruct_variable_diffusivity_profile(
        z,
        lambda_anode=lambda_anode,
        lambda_cathode=lambda_cathode,
        flux_lambda_m_s=flux_lambda_m_s,
        drag_velocity_m_s=drag_velocity_m_s,
        diffusivity_model=diffusivity_model,
        quadrature_points=quadrature_points,
    )
    profile[-1] = lambda_cathode
    return profile


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
        lambda_lower_bound=1.0e-4,
        lambda_upper_bound=16.99,
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
