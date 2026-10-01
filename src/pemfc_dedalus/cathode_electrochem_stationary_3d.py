"""Stationary nonlinear 3D cathode electrochemistry solver for V1.0."""

from __future__ import annotations

from typing import TypedDict

import dedalus.public as d3
import numpy as np

from .parameters import CathodeParameters


class ContinuationStage(TypedDict):
    reaction_scale: float
    iterations: int
    perturbation_norm: float
    converged: bool
    perturbation_norm_history: list[float]
    min_c_o2_history: list[float]
    component_norm_history: list[dict[str, float]]
    physical_norm_history: list[float]


class StationaryResult(TypedDict):
    converged: bool
    newton_iterations: int
    perturbation_norm: float
    newton_damping: float
    continuation_history: list[ContinuationStage]
    total_reaction_current: float
    mean_eta_v: float
    mean_c_o2_mol_m3: float
    min_c_o2_mol_m3: float
    min_eta_v: float
    max_eta_v: float
    min_j_orr_a_m3: float
    max_j_orr_a_m3: float
    initial_min_c_o2_mol_m3: float
    initial_max_c_o2_mol_m3: float
    initial_mean_c_o2_mol_m3: float


def solve_stationary(
    *,
    params: CathodeParameters,
    nx: int = 8,
    ny: int = 8,
    nz: int = 32,
    oxygen_feed_concentration: float | None = None,
    newton_tolerance: float = 1e-8,
    max_newton_iterations: int = 30,
    newton_damping: float = 0.25,
    reaction_scales: tuple[float, ...] = (
        1e-6,
        1e-5,
        1e-4,
        2e-4,
        3e-4,
        5e-4,
        7e-4,
        1e-3,
    ),
) -> StationaryResult:
    if not 0.0 < newton_damping <= 1.0:
        raise ValueError("newton_damping must be in (0, 1]")
    if not reaction_scales:
        raise ValueError("reaction_scales must be non-empty")
    if any(scale <= 0.0 or scale > 1.0 for scale in reaction_scales):
        raise ValueError("reaction scales must lie in (0, 1]")

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

    porosity = dist.Field(name="porosity", bases=zbasis)
    porosity["g"] = (
        params.porosity_gdl * (1.0 - chi_cl["g"])
        + params.porosity_cl * chi_cl["g"]
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
    beta_a = params.beta_anodic
    beta_c = params.beta_cathodic
    j0_vol = params.j0_vol
    gamma_o2 = params.oxygen_reaction_order
    phi_s_bc = params.cathode_solid_potential
    phi_m_bc = params.membrane_proton_potential
    Lz = params.thickness_z

    eta = phi_s - phi_m - E_eq
    oxygen_activity = c / c_ref
    bv_exp_limit = 40.0
    cathodic_arg_raw = -beta_c * eta
    anodic_arg_raw = beta_a * eta
    cathodic_arg = bv_exp_limit * np.tanh(cathodic_arg_raw / bv_exp_limit)
    anodic_arg = bv_exp_limit * np.tanh(anodic_arg_raw / bv_exp_limit)

    reaction_scale = dist.Field(name="reaction_scale")
    reaction_scale["g"] = reaction_scales[0]

    # In the intended cathodic operating branch eta is negative and the net
    # Butler-Volmer source is positive. Avoid abs() here so the NLBVP has a
    # differentiable source for Newton linearisation.
    j_orr = (
        reaction_scale
        * chi_cl
        * j0_vol
        * oxygen_activity**gamma_o2
        * (np.exp(cathodic_arg) - np.exp(anodic_arg))
    )
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
    linear_solver = linear_problem.build_solver()
    linear_solver.solve()

    c.change_scales(1)
    initial_c_values = np.asarray(c["g"]).copy()
    initial_min_c_o2 = float(np.min(initial_c_values))
    initial_max_c_o2 = float(np.max(initial_c_values))
    initial_mean_c_o2 = float(np.mean(initial_c_values))

    problem = d3.NLBVP(
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

    problem.add_equation(
        "-div(diffusivity*grad_c) + lift(tau_c2, -1) = -s_o2"
    )
    problem.add_equation(
        "-div(sigma_s*grad_phi_s) + lift(tau_s2, -1) = -j_orr"
    )
    problem.add_equation(
        "-div(sigma_m*grad_phi_m) + lift(tau_m2, -1) = -j_orr"
    )

    problem.add_equation("c(z=0) = inlet")
    problem.add_equation("ez @ grad_c(z=Lz) = 0")
    problem.add_equation("phi_s(z=0) = phi_s_bc")
    problem.add_equation("ez @ grad_phi_s(z=Lz) = 0")
    problem.add_equation("ez @ grad_phi_m(z=0) = 0")
    problem.add_equation("phi_m(z=Lz) = phi_m_bc")

    solver = problem.build_solver()

    perturbation_norm = np.inf
    total_iterations = 0
    continuation_history: list[ContinuationStage] = [
        {
            "reaction_scale": 0.0,
            "iterations": 1,
            "perturbation_norm": 0.0,
            "converged": True,
            "perturbation_norm_history": [0.0],
            "min_c_o2_history": [initial_min_c_o2],
            "component_norm_history": [],
            "physical_norm_history": [0.0],
        }
    ]

    for scale in reaction_scales:
        reaction_scale["g"] = scale
        stage_iterations = 0
        perturbation_norm = np.inf
        perturbation_norm_history: list[float] = []
        min_c_o2_history: list[float] = []
        component_norm_history: list[dict[str, float]] = []
        physical_norm_history: list[float] = []

        stage_damping = newton_damping

        physical_norm = np.inf
        while (
            stage_iterations < max_newton_iterations
            and physical_norm > newton_tolerance
        ):
            solver.newton_iteration(damping=stage_damping)
            perturbation_norm = float(
                sum(
                    perturbation.allreduce_data_norm("c", 2)
                    for perturbation in solver.perturbations
                )
            )
            perturbation_norm_history.append(perturbation_norm)
            component_norms: dict[str, float] = {}
            for index, perturbation in enumerate(solver.perturbations):
                name = perturbation.name or f"perturbation_{index}"
                component_norms[name] = float(
                    perturbation.allreduce_data_norm("c", 2)
                )
            component_norm_history.append(component_norms)
            physical_norm = float(
                sum(
                    perturbation.allreduce_data_norm("c", 2)
                    for perturbation in solver.perturbations[:3]
                )
            )
            physical_norm_history.append(physical_norm)
            c.change_scales(1)
            min_c_o2_history.append(float(np.min(np.asarray(c["g"]))))
            stage_iterations += 1
            total_iterations += 1

        stage_converged = physical_norm <= newton_tolerance
        continuation_history.append(
            {
                "reaction_scale": scale,
                "iterations": stage_iterations,
                "perturbation_norm": perturbation_norm,
                "converged": stage_converged,
                "perturbation_norm_history": perturbation_norm_history,
                "min_c_o2_history": min_c_o2_history,
                "component_norm_history": component_norm_history,
                "physical_norm_history": physical_norm_history,
            }
        )
        if not stage_converged:
            break

    c.change_scales(1)
    phi_s.change_scales(1)
    phi_m.change_scales(1)

    c_values = np.asarray(c["g"]).copy()
    eta_values = np.asarray((phi_s - phi_m - E_eq).evaluate()["g"]).copy()
    j_values = np.asarray(j_orr.evaluate()["g"]).copy()

    total_reaction_current = float(d3.Integrate(j_orr).evaluate()["g"].ravel()[0])
    mean_eta = float(d3.Average(eta).evaluate()["g"].ravel()[0])
    mean_c_o2 = float(d3.Average(c).evaluate()["g"].ravel()[0])

    return {
        "converged": (
            len(continuation_history) == len(reaction_scales) + 1
            and bool(continuation_history[-1]["converged"])
        ),
        "newton_iterations": total_iterations,
        "perturbation_norm": perturbation_norm,
        "newton_damping": newton_damping,
        "continuation_history": continuation_history,
        "total_reaction_current": total_reaction_current,
        "mean_eta_v": mean_eta,
        "mean_c_o2_mol_m3": mean_c_o2,
        "min_c_o2_mol_m3": float(np.min(c_values)),
        "min_eta_v": float(np.min(eta_values)),
        "max_eta_v": float(np.max(eta_values)),
        "min_j_orr_a_m3": float(np.min(j_values)),
        "max_j_orr_a_m3": float(np.max(j_values)),
        "initial_min_c_o2_mol_m3": initial_min_c_o2,
        "initial_max_c_o2_mol_m3": initial_max_c_o2,
        "initial_mean_c_o2_mol_m3": initial_mean_c_o2,
    }
