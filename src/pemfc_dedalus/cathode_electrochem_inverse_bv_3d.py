"""Stationary 3D cathode solve with exact inverse symmetric Butler-Volmer kinetics."""

from __future__ import annotations

from typing import TypedDict

import dedalus.public as d3
import numpy as np

from .parameters import CathodeParameters


class InverseBVResult(TypedDict):
    converged: bool
    newton_iterations: int
    physical_norm: float
    physical_norm_history: list[float]
    min_c_o2_history: list[float]
    component_norm_history: list[dict[str, float]]
    total_reaction_current: float
    mean_eta_v: float
    mean_c_o2_mol_m3: float
    min_c_o2_mol_m3: float
    min_eta_v: float
    max_eta_v: float
    min_q_hat: float
    max_q_hat: float
    min_q_a_m3: float
    max_q_a_m3: float
    q_reference_a_m3: float
    max_inverse_bv_residual_v: float
    max_forward_bv_relative_error: float
    initial_min_c_o2_mol_m3: float
    initial_max_c_o2_mol_m3: float
    initial_mean_c_o2_mol_m3: float


def symmetric_bv_current(
    eta_v: float,
    exchange_current: float,
    beta_per_v: float,
) -> float:
    """Return cathodic-positive current for symmetric Butler-Volmer kinetics."""
    return float(2.0 * exchange_current * np.sinh(-beta_per_v * eta_v))


def inverse_symmetric_bv_eta(
    current: float,
    exchange_current: float,
    beta_per_v: float,
) -> float:
    """Invert symmetric Butler-Volmer exactly for the overpotential."""
    if exchange_current <= 0.0:
        raise ValueError("exchange_current must be positive")
    if beta_per_v <= 0.0:
        raise ValueError("beta_per_v must be positive")
    return float(
        -np.arcsinh(current / (2.0 * exchange_current)) / beta_per_v
    )


def solve_stationary_inverse_bv(
    *,
    params: CathodeParameters,
    nx: int = 8,
    ny: int = 8,
    nz: int = 32,
    oxygen_feed_concentration: float | None = None,
    newton_tolerance: float = 1e-7,
    max_newton_iterations: int = 30,
    newton_damping: float = 1.0,
) -> InverseBVResult:
    """Solve the stationary cathode with q as an algebraic inverse-BV unknown."""
    if not 0.0 < newton_damping <= 1.0:
        raise ValueError("newton_damping must be in (0, 1]")
    if newton_tolerance <= 0.0:
        raise ValueError("newton_tolerance must be positive")
    if max_newton_iterations < 1:
        raise ValueError("max_newton_iterations must be positive")

    beta_a = params.beta_anodic
    beta_c = params.beta_cathodic
    if not np.isclose(beta_a, beta_c, rtol=1e-12, atol=0.0):
        raise ValueError(
            "inverse symmetric Butler-Volmer requires beta_anodic == beta_cathodic"
        )
    beta = 0.5 * (beta_a + beta_c)

    cl_thickness = params.thickness_z - params.gdl_thickness
    if cl_thickness <= 0.0:
        raise ValueError("catalyst-layer thickness must be positive")
    q_reference = params.membrane_current_density / cl_thickness

    coords = d3.CartesianCoordinates("x", "y", "z")
    dist = d3.Distributor(coords, dtype=np.float64)

    xbasis = d3.RealFourier(
        coords["x"], size=nx, bounds=(0.0, params.length_x), dealias=3 / 2
    )
    ybasis = d3.RealFourier(
        coords["y"], size=ny, bounds=(0.0, params.length_y), dealias=3 / 2
    )
    zbasis = d3.ChebyshevT(
        coords["z"], size=nz, bounds=(0.0, params.thickness_z), dealias=3 / 2
    )
    bases = (xbasis, ybasis, zbasis)

    c = dist.Field(name="c_o2", bases=bases)
    phi_s = dist.Field(name="phi_s", bases=bases)
    phi_m = dist.Field(name="phi_m", bases=bases)
    q_hat = dist.Field(name="q_hat", bases=bases)

    tau_c1 = dist.Field(name="tau_c1", bases=(xbasis, ybasis))
    tau_c2 = dist.Field(name="tau_c2", bases=(xbasis, ybasis))
    tau_s1 = dist.Field(name="tau_s1", bases=(xbasis, ybasis))
    tau_s2 = dist.Field(name="tau_s2", bases=(xbasis, ybasis))
    tau_m1 = dist.Field(name="tau_m1", bases=(xbasis, ybasis))
    tau_m2 = dist.Field(name="tau_m2", bases=(xbasis, ybasis))

    z1 = dist.local_grid(zbasis)

    chi_cl = dist.Field(name="chi_cl", bases=zbasis)
    chi_cl["g"] = 0.5 * (
        1.0 + np.tanh((z1 - params.gdl_thickness) / params.interface_width)
    )

    diffusivity = dist.Field(name="d_eff", bases=zbasis)
    diffusivity["g"] = (
        params.d_o2_gdl * (1.0 - chi_cl["g"])
        + params.d_o2_cl * chi_cl["g"]
    )

    sigma_s = dist.Field(name="sigma_s", bases=zbasis)
    sigma_s["g"] = (
        params.sigma_s_gdl * (1.0 - chi_cl["g"])
        + params.sigma_s_cl * chi_cl["g"]
    )

    sigma_m = dist.Field(name="sigma_m", bases=zbasis)
    sigma_m["g"] = (
        params.sigma_m_floor * (1.0 - chi_cl["g"])
        + params.sigma_m_cl * chi_cl["g"]
    )

    c_ref = params.oxygen_inlet_concentration
    c_feed = c_ref if oxygen_feed_concentration is None else oxygen_feed_concentration
    if c_feed <= 0.0:
        raise ValueError("oxygen_feed_concentration must be positive")

    inlet = dist.Field(name="c_inlet", bases=(xbasis, ybasis))
    x2, y2 = dist.local_grids(xbasis, ybasis)
    inlet["g"] = c_feed * (
        0.92
        + 0.08
        * np.cos(2.0 * np.pi * y2 / params.length_y)
        * (0.95 + 0.05 * np.cos(2.0 * np.pi * x2 / params.length_x))
    )

    c["g"] = c_feed
    phi_s["g"] = params.cathode_solid_potential
    phi_m["g"] = params.membrane_proton_potential
    q_hat["g"] = 0.0

    grad = d3.grad
    div = d3.div
    lift_basis = zbasis.derivative_basis(1)

    def lift(field, n):
        return d3.Lift(field, lift_basis, n)

    ez = coords.unit_vector_fields(dist)[2]
    grad_c = grad(c) + ez * lift(tau_c1, -1)
    grad_phi_s = grad(phi_s) + ez * lift(tau_s1, -1)
    grad_phi_m = grad(phi_m) + ez * lift(tau_m1, -1)

    F = params.faraday
    E_eq = params.equilibrium_potential
    j0_vol = params.j0_vol
    gamma_o2 = params.oxygen_reaction_order
    phi_s_bc = params.cathode_solid_potential
    phi_m_bc = params.membrane_proton_potential
    Lz = params.thickness_z

    eta = phi_s - phi_m - E_eq
    oxygen_activity = c / c_ref
    q = q_reference * q_hat
    exchange_current = j0_vol * oxygen_activity**gamma_o2
    inverse_bv_residual = (
        eta + np.arcsinh(q / (2.0 * exchange_current)) / beta
    )
    j_orr = chi_cl * q
    s_o2 = j_orr / (4.0 * F)

    linear_problem = d3.LBVP(
        [
            c,
            phi_s,
            phi_m,
            tau_c1,
            tau_c2,
            tau_s1,
            tau_s2,
            tau_m1,
            tau_m2,
        ],
        namespace=locals(),
    )
    linear_problem.add_equation(
        "-div(diffusivity*grad_c) + lift(tau_c2, -1) = 0"
    )
    linear_problem.add_equation(
        "-div(sigma_s*grad_phi_s) + lift(tau_s2, -1) = 0"
    )
    linear_problem.add_equation(
        "-div(sigma_m*grad_phi_m) + lift(tau_m2, -1) = 0"
    )
    linear_problem.add_equation("c(z=0) = inlet")
    linear_problem.add_equation("ez @ grad_c(z=Lz) = 0")
    linear_problem.add_equation("phi_s(z=0) = phi_s_bc")
    linear_problem.add_equation("ez @ grad_phi_s(z=Lz) = 0")
    linear_problem.add_equation("ez @ grad_phi_m(z=0) = 0")
    linear_problem.add_equation("phi_m(z=Lz) = phi_m_bc")
    linear_problem.build_solver().solve()

    c.change_scales(1)
    phi_s.change_scales(1)
    phi_m.change_scales(1)
    initial_c_values = np.asarray(c["g"]).copy()
    initial_min_c_o2 = float(np.min(initial_c_values))
    initial_max_c_o2 = float(np.max(initial_c_values))
    initial_mean_c_o2 = float(np.mean(initial_c_values))

    eta_initial = np.asarray(eta.evaluate()["g"])
    activity_initial = np.asarray(c["g"]) / c_ref
    exchange_initial = j0_vol * activity_initial**gamma_o2
    q_hat["g"] = (
        2.0 * exchange_initial * np.sinh(-beta * eta_initial) / q_reference
    )

    problem = d3.NLBVP(
        [
            c,
            phi_s,
            phi_m,
            q_hat,
            tau_c1,
            tau_c2,
            tau_s1,
            tau_s2,
            tau_m1,
            tau_m2,
        ],
        namespace=locals(),
    )
    problem.add_equation(
        "-div(diffusivity*grad_c) + lift(tau_c2, -1) = -s_o2"
    )
    problem.add_equation(
        "-div(sigma_s*grad_phi_s) + lift(tau_s2, -1) = -j_orr"
    )
    problem.add_equation(
        "-div(sigma_m*grad_phi_m) + lift(tau_m2, -1) = -j_orr"
    )
    problem.add_equation("inverse_bv_residual = 0")
    problem.add_equation("c(z=0) = inlet")
    problem.add_equation("ez @ grad_c(z=Lz) = 0")
    problem.add_equation("phi_s(z=0) = phi_s_bc")
    problem.add_equation("ez @ grad_phi_s(z=Lz) = 0")
    problem.add_equation("ez @ grad_phi_m(z=0) = 0")
    problem.add_equation("phi_m(z=Lz) = phi_m_bc")

    solver = problem.build_solver()

    physical_norm = np.inf
    physical_norm_history: list[float] = []
    min_c_o2_history: list[float] = []
    component_norm_history: list[dict[str, float]] = []
    iterations = 0

    while iterations < max_newton_iterations and physical_norm > newton_tolerance:
        solver.newton_iteration(damping=newton_damping)
        component_norms: dict[str, float] = {}
        for index, perturbation in enumerate(solver.perturbations[:4]):
            name = perturbation.name or f"perturbation_{index}"
            component_norms[name] = float(
                perturbation.allreduce_data_norm("c", 2)
            )
        component_norm_history.append(component_norms)
        physical_norm = float(sum(component_norms.values()))
        physical_norm_history.append(physical_norm)

        c.change_scales(1)
        c_min = float(np.min(np.asarray(c["g"])))
        min_c_o2_history.append(c_min)
        iterations += 1

        if not np.isfinite(physical_norm) or c_min <= 0.0:
            break

    converged = (
        np.isfinite(physical_norm)
        and physical_norm <= newton_tolerance
        and bool(min_c_o2_history)
        and min_c_o2_history[-1] > 0.0
    )

    c.change_scales(1)
    phi_s.change_scales(1)
    phi_m.change_scales(1)
    q_hat.change_scales(1)

    c_values = np.asarray(c["g"]).copy()
    eta_values = np.asarray(eta.evaluate()["g"]).copy()
    q_hat_values = np.asarray(q_hat["g"]).copy()
    q_values = q_reference * q_hat_values
    exchange_values = j0_vol * (c_values / c_ref) ** gamma_o2

    inverse_residual_values = (
        eta_values
        + np.arcsinh(q_values / (2.0 * exchange_values)) / beta
    )
    forward_values = (
        2.0 * exchange_values * np.sinh(-beta * eta_values)
    )
    forward_scale = np.maximum(np.abs(q_values), q_reference * 1e-12)
    forward_relative_error = np.abs(forward_values - q_values) / forward_scale

    total_reaction_current = float(d3.Integrate(j_orr).evaluate()["g"].ravel()[0])
    mean_eta = float(d3.Average(eta).evaluate()["g"].ravel()[0])
    mean_c_o2 = float(d3.Average(c).evaluate()["g"].ravel()[0])

    return {
        "converged": converged,
        "newton_iterations": iterations,
        "physical_norm": physical_norm,
        "physical_norm_history": physical_norm_history,
        "min_c_o2_history": min_c_o2_history,
        "component_norm_history": component_norm_history,
        "total_reaction_current": total_reaction_current,
        "mean_eta_v": mean_eta,
        "mean_c_o2_mol_m3": mean_c_o2,
        "min_c_o2_mol_m3": float(np.min(c_values)),
        "min_eta_v": float(np.min(eta_values)),
        "max_eta_v": float(np.max(eta_values)),
        "min_q_hat": float(np.min(q_hat_values)),
        "max_q_hat": float(np.max(q_hat_values)),
        "min_q_a_m3": float(np.min(q_values)),
        "max_q_a_m3": float(np.max(q_values)),
        "q_reference_a_m3": float(q_reference),
        "max_inverse_bv_residual_v": float(np.max(np.abs(inverse_residual_values))),
        "max_forward_bv_relative_error": float(np.max(forward_relative_error)),
        "initial_min_c_o2_mol_m3": initial_min_c_o2,
        "initial_max_c_o2_mol_m3": initial_max_c_o2,
        "initial_mean_c_o2_mol_m3": initial_mean_c_o2,
    }
