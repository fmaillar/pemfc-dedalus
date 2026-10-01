"""Stationary 3D cathode solve with exact inverse symmetric Butler-Volmer kinetics."""

from __future__ import annotations

from typing import TypedDict

import dedalus.public as d3
import numpy as np

from .parameters import CathodeParameters


class InverseBVBacktrack(TypedDict):
    iteration: int
    residual_start: float
    residual_final: float
    damping: float
    backtracks: int
    min_c_o2_mol_m3: float
    min_q_hat: float


class InverseBVPicardStep(TypedDict):
    iteration: int
    min_c_o2_mol_m3: float
    min_q_hat: float
    max_q_hat: float
    relative_q_change: float


class InverseBVResidualComponents(TypedDict):
    oxygen: float
    solid_potential: float
    membrane_potential: float
    inverse_bv: float
    bc_c_inlet: float
    bc_c_flux: float
    bc_phi_s: float
    bc_phi_s_flux: float
    bc_phi_m_flux: float
    bc_phi_m: float


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
    globalized: bool
    residual_merit: float
    globalization_history: list[InverseBVBacktrack]
    picard_iterations: int
    picard_relaxation: float
    picard_history: list[InverseBVPicardStep]
    residual_components: InverseBVResidualComponents


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
    globalized: bool = False,
    residual_tolerance: float = 1e-8,
    max_backtracks: int = 10,
    picard_iterations: int = 0,
    picard_relaxation: float = 0.25,
    picard_tolerance: float = 0.0,
) -> InverseBVResult:
    """Solve the stationary cathode with q as an algebraic inverse-BV unknown."""
    if not 0.0 < newton_damping <= 1.0:
        raise ValueError("newton_damping must be in (0, 1]")
    if newton_tolerance <= 0.0:
        raise ValueError("newton_tolerance must be positive")
    if max_newton_iterations < 1:
        raise ValueError("max_newton_iterations must be positive")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")
    if max_backtracks < 1:
        raise ValueError("max_backtracks must be positive")
    if picard_iterations < 0:
        raise ValueError("picard_iterations must be non-negative")
    if not 0.0 < picard_relaxation <= 1.0:
        raise ValueError("picard_relaxation must be in (0, 1]")
    if picard_tolerance < 0.0:
        raise ValueError("picard_tolerance must be non-negative")

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

    eta_initial_field = eta.evaluate()
    eta_initial_field.change_scales(1)
    eta_initial = np.asarray(eta_initial_field["g"]).copy()
    activity_initial = np.asarray(c["g"]) / c_ref
    exchange_initial = j0_vol * activity_initial**gamma_o2
    q_hat["g"] = (
        2.0 * exchange_initial * np.sinh(-beta * eta_initial) / q_reference
    )

    picard_history: list[InverseBVPicardStep] = []
    if picard_iterations:
        picard_problem = d3.LBVP(
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
        picard_problem.add_equation(
            "-div(diffusivity*grad_c) + lift(tau_c2, -1) = -s_o2"
        )
        picard_problem.add_equation(
            "-div(sigma_s*grad_phi_s) + lift(tau_s2, -1) = -j_orr"
        )
        picard_problem.add_equation(
            "-div(sigma_m*grad_phi_m) + lift(tau_m2, -1) = -j_orr"
        )
        picard_problem.add_equation("c(z=0) = inlet")
        picard_problem.add_equation("ez @ grad_c(z=Lz) = 0")
        picard_problem.add_equation("phi_s(z=0) = phi_s_bc")
        picard_problem.add_equation("ez @ grad_phi_s(z=Lz) = 0")
        picard_problem.add_equation("ez @ grad_phi_m(z=0) = 0")
        picard_problem.add_equation("phi_m(z=Lz) = phi_m_bc")
        picard_solver = picard_problem.build_solver()

        for picard_iteration in range(1, picard_iterations + 1):
            q_hat.change_scales(1)
            old_q_hat = np.asarray(q_hat["g"]).copy()

            picard_solver.solve()

            c.change_scales(1)
            eta_picard_field = eta.evaluate()
            eta_picard_field.change_scales(1)
            eta_picard = np.asarray(eta_picard_field["g"]).copy()
            c_picard = np.asarray(c["g"]).copy()
            exchange_picard = j0_vol * (c_picard / c_ref) ** gamma_o2
            q_bv_hat = (
                2.0
                * exchange_picard
                * np.sinh(-beta * eta_picard)
                / q_reference
            )

            if (
                not np.all(np.isfinite(q_bv_hat))
                or float(np.min(c_picard)) <= 0.0
                or float(np.min(q_bv_hat)) <= 0.0
            ):
                break

            relaxed_q_hat = (
                (1.0 - picard_relaxation) * old_q_hat
                + picard_relaxation * q_bv_hat
            )
            q_scale = np.maximum(np.abs(old_q_hat), 1e-12)
            relative_q_change = float(
                np.max(np.abs(relaxed_q_hat - old_q_hat) / q_scale)
            )
            q_hat.change_scales(1)
            q_hat["g"] = relaxed_q_hat

            picard_history.append(
                {
                    "iteration": picard_iteration,
                    "min_c_o2_mol_m3": float(np.min(c_picard)),
                    "min_q_hat": float(np.min(relaxed_q_hat)),
                    "max_q_hat": float(np.max(relaxed_q_hat)),
                    "relative_q_change": relative_q_change,
                }
            )

            if (
                picard_tolerance > 0.0
                and relative_q_change <= picard_tolerance
            ):
                break

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

    residual_expressions = (
        -div(diffusivity * grad_c) + lift(tau_c2, -1) + s_o2,
        -div(sigma_s * grad_phi_s) + lift(tau_s2, -1) + j_orr,
        -div(sigma_m * grad_phi_m) + lift(tau_m2, -1) + j_orr,
        inverse_bv_residual,
        c(z=0) - inlet,
        ez @ grad_c(z=Lz),
        phi_s(z=0) - phi_s_bc,
        ez @ grad_phi_s(z=Lz),
        ez @ grad_phi_m(z=0),
        phi_m(z=Lz) - phi_m_bc,
    )
    residual_scales = (
        max(params.d_o2_gdl, params.d_o2_cl) * c_ref / Lz**2,
        max(params.sigma_s_gdl, params.sigma_s_cl) / Lz**2,
        max(params.sigma_m_floor, params.sigma_m_cl) / Lz**2,
        1.0,
        c_ref,
        c_ref / Lz,
        1.0,
        1.0 / Lz,
        1.0 / Lz,
        1.0,
    )

    residual_names = (
        "oxygen",
        "solid_potential",
        "membrane_potential",
        "inverse_bv",
        "bc_c_inlet",
        "bc_c_flux",
        "bc_phi_s",
        "bc_phi_s_flux",
        "bc_phi_m_flux",
        "bc_phi_m",
    )

    def normalized_residuals() -> list[float]:
        values: list[float] = []
        for expression, scale in zip(
            residual_expressions,
            residual_scales,
            strict=True,
        ):
            residual_field = expression.evaluate()
            values.append(
                float(residual_field.allreduce_data_norm("c", 2)) / scale
            )
        return values

    def residual_merit() -> float:
        normalized = normalized_residuals()
        return float(np.sqrt(sum(value * value for value in normalized)))

    def snapshot_state() -> list[np.ndarray]:
        snapshot: list[np.ndarray] = []
        for field in solver.state:
            field.change_layout("c")
            snapshot.append(np.asarray(field["c"]).copy())
        return snapshot

    def restore_state(snapshot: list[np.ndarray]) -> None:
        for field, data in zip(solver.state, snapshot, strict=True):
            field.change_layout("c")
            field["c"] = data

    def physical_bounds() -> tuple[float, float]:
        c.change_scales(1)
        q_hat.change_scales(1)
        return (
            float(np.min(np.asarray(c["g"]))),
            float(np.min(np.asarray(q_hat["g"]))),
        )

    physical_norm = np.inf
    physical_norm_history: list[float] = []
    min_c_o2_history: list[float] = []
    component_norm_history: list[dict[str, float]] = []
    globalization_history: list[InverseBVBacktrack] = []
    iterations = 0
    final_residual_merit = residual_merit()

    if globalized:
        while (
            iterations < max_newton_iterations
            and final_residual_merit > residual_tolerance
        ):
            base_state = snapshot_state()
            base_iteration = solver.iteration
            residual_start = final_residual_merit
            damping = 1.0
            accepted = False
            used_backtracks = 0

            for backtrack in range(max_backtracks):
                restore_state(base_state)
                solver.iteration = base_iteration
                solver.newton_iteration(damping=damping)
                trial_residual = residual_merit()
                c_min, q_min = physical_bounds()

                if (
                    np.isfinite(trial_residual)
                    and trial_residual < residual_start
                    and c_min > 0.0
                    and q_min > 0.0
                ):
                    accepted = True
                    used_backtracks = backtrack
                    final_residual_merit = trial_residual
                    break

                damping *= 0.5

            if not accepted:
                restore_state(base_state)
                solver.iteration = base_iteration
                break

            component_norms: dict[str, float] = {}
            for index, perturbation in enumerate(solver.perturbations[:4]):
                name = perturbation.name or f"perturbation_{index}"
                component_norms[name] = float(
                    perturbation.allreduce_data_norm("c", 2)
                )
            component_norm_history.append(component_norms)
            physical_norm = float(sum(component_norms.values()))
            physical_norm_history.append(physical_norm)
            c_min, q_min = physical_bounds()
            min_c_o2_history.append(c_min)
            iterations += 1
            globalization_history.append(
                {
                    "iteration": iterations,
                    "residual_start": residual_start,
                    "residual_final": final_residual_merit,
                    "damping": damping,
                    "backtracks": used_backtracks,
                    "min_c_o2_mol_m3": c_min,
                    "min_q_hat": q_min,
                }
            )

        c_min, q_min = physical_bounds()
        converged = (
            np.isfinite(final_residual_merit)
            and final_residual_merit <= residual_tolerance
            and c_min > 0.0
            and q_min > 0.0
        )
    else:
        while (
            iterations < max_newton_iterations
            and physical_norm > newton_tolerance
        ):
            solver.newton_iteration(damping=newton_damping)
            component_norms = {}
            for index, perturbation in enumerate(solver.perturbations[:4]):
                name = perturbation.name or f"perturbation_{index}"
                component_norms[name] = float(
                    perturbation.allreduce_data_norm("c", 2)
                )
            component_norm_history.append(component_norms)
            physical_norm = float(sum(component_norms.values()))
            physical_norm_history.append(physical_norm)

            c_min, _ = physical_bounds()
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
        final_residual_merit = residual_merit()

    c.change_scales(1)
    phi_s.change_scales(1)
    phi_m.change_scales(1)
    q_hat.change_scales(1)

    c_values = np.asarray(c["g"]).copy()
    eta_field = eta.evaluate()
    eta_field.change_scales(1)
    eta_values = np.asarray(eta_field["g"]).copy()
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

    residual_component_values = normalized_residuals()
    residual_components: InverseBVResidualComponents = {
        "oxygen": residual_component_values[0],
        "solid_potential": residual_component_values[1],
        "membrane_potential": residual_component_values[2],
        "inverse_bv": residual_component_values[3],
        "bc_c_inlet": residual_component_values[4],
        "bc_c_flux": residual_component_values[5],
        "bc_phi_s": residual_component_values[6],
        "bc_phi_s_flux": residual_component_values[7],
        "bc_phi_m_flux": residual_component_values[8],
        "bc_phi_m": residual_component_values[9],
    }

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
        "globalized": globalized,
        "residual_merit": final_residual_merit,
        "globalization_history": globalization_history,
        "picard_iterations": len(picard_history),
        "picard_relaxation": picard_relaxation,
        "picard_history": picard_history,
        "residual_components": residual_components,
    }
