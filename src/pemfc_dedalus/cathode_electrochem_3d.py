"""3D open-cathode PEMFC cathode electrochemistry validation model.

V0.2 couples:
  * O2 diffusion in cathode GDL + catalyst layer,
  * electronic conduction,
  * protonic conduction in the catalyst layer,
  * concentration-dependent Butler-Volmer ORR kinetics.

The cathode is treated as open to an external air stream.  The air-feed
boundary supplies oxygen; forced-air momentum and heat transfer are added in a
later model.  The anode is intentionally absent here; the target architecture
is dead-end H2 with purge events, which will be added after the MEA cathode and
membrane models are validated.

The electrical equations are advanced in pseudo-time.  Their converged state is
the stationary conduction solution; the pseudo-time itself is not yet intended
to represent a physical double-layer transient.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import dedalus.public as d3
import numpy as np

from .parameters import CathodeParameters

logger = logging.getLogger(__name__)


def build_solver(
    params: CathodeParameters,
    *,
    nx: int,
    ny: int,
    nz: int,
    stop_time: float,
    output_dir: Path,
):
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

    x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
    z1 = dist.local_grid(zbasis)

    chi_cl = dist.Field(name="chi_cl", bases=zbasis)
    chi_cl["g"] = 0.5 * (
        1.0
        + np.tanh(
            (z1 - params.gdl_thickness) / params.interface_width
        )
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

    # Open-cathode air-feed boundary.  The smooth x/y modulation represents
    # nonuniform external airflow / rib exposure while retaining periodic bases.
    c_ref = params.oxygen_inlet_concentration
    inlet = dist.Field(name="c_inlet", bases=(xbasis, ybasis))
    x2, y2 = dist.local_grids(xbasis, ybasis)
    inlet["g"] = c_ref * (
        0.92
        + 0.08
        * np.cos(2.0 * np.pi * y2 / params.length_y)
        * (0.95 + 0.05 * np.cos(2.0 * np.pi * x2 / params.length_x))
    )

    # Initial guesses.
    c["g"] = c_ref
    phi_s["g"] = params.cathode_solid_potential
    phi_m["g"] = params.membrane_proton_potential

    grad = d3.grad
    div = d3.div

    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return d3.Lift(A, lift_basis, n)
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
    C_s = params.pseudo_capacitance_s
    C_m = params.pseudo_capacitance_m
    phi_s_bc = params.cathode_solid_potential
    phi_m_bc = params.membrane_proton_potential
    Lz = params.thickness_z

    # eta < 0 for cathodic operation. j_orr is defined positive for ORR.
    #
    # During pseudo-transient continuation, the potential fields can briefly
    # overshoot. Raw Butler-Volmer exponentials can then overflow long before
    # the physical steady state is reached. We therefore use a smooth limiter
    # on the exponent arguments. It is effectively identity in the physical
    # operating range, but asymptotes before floating-point overflow.
    eta = phi_s - phi_m - E_eq
    oxygen_activity = c / c_ref
    bv_exp_limit = 40.0
    cathodic_arg_raw = -beta_c * eta
    anodic_arg_raw = beta_a * eta
    cathodic_arg = bv_exp_limit * np.tanh(cathodic_arg_raw / bv_exp_limit)
    anodic_arg = bv_exp_limit * np.tanh(anodic_arg_raw / bv_exp_limit)
    j_bv_net = (
        j0_vol
        * oxygen_activity**gamma_o2
        * (
            np.exp(cathodic_arg)
            - np.exp(anodic_arg)
        )
    )

    # This cathode model represents oxygen reduction only. A transient local
    # positive eta can make the reversible Butler-Volmer expression negative,
    # which would imply oxygen generation. Project the net current smoothly
    # onto its positive (ORR) branch.
    j_positive_eps = 1e-6 * j0_vol
    j_orr = chi_cl * 0.5 * (
        j_bv_net + np.sqrt(j_bv_net**2 + j_positive_eps**2)
    )
    s_o2 = j_orr / (4.0 * F)

    problem = d3.IVP(
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

    # Oxygen balance.
    problem.add_equation(
        "porosity*dt(c) - div(diffusivity*grad_c) + lift(tau_c2, -1) = -s_o2"
    )

    # Solid-potential pseudo-transient. Steady state:
    # div(sigma_s grad(phi_s)) = j_orr.
    # Keep the full variable-coefficient conduction operator implicit; an
    # explicit conductivity residual is far too stiff for the CL/GDL contrast.
    problem.add_equation(
        "C_s*dt(phi_s) - div(sigma_s*grad_phi_s) + lift(tau_s2, -1) = -j_orr"
    )

    # Protonic-potential pseudo-transient. Steady state:
    # div(sigma_m grad(phi_m)) = -j_orr.
    # The full sigma_m operator is implicit for the same stiffness reason.
    problem.add_equation(
        "C_m*dt(phi_m) - div(sigma_m*grad_phi_m) + lift(tau_m2, -1) = j_orr"
    )

    # O2: prescribed open-air feed at z=0; no O2 penetration into membrane.
    problem.add_equation("c(z=0) = inlet")
    problem.add_equation("ez @ grad_c(z=Lz) = 0")

    # Electrons enter from the cathode current-collector/GDL side and do not
    # cross into the membrane.
    problem.add_equation("phi_s(z=0) = phi_s_bc")
    problem.add_equation("ez @ grad_phi_s(z=Lz) = 0")

    # Protons do not enter the gas side; membrane interface fixes reference.
    problem.add_equation("ez @ grad_phi_m(z=0) = 0")
    problem.add_equation("phi_m(z=Lz) = phi_m_bc")

    solver = problem.build_solver(d3.SBDF2)
    solver.stop_sim_time = stop_time

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = solver.evaluator.add_file_handler(
        str(output_dir / "snapshots"),
        sim_dt=max(stop_time / 10.0, 1e-7),
        max_writes=20,
    )
    snapshots.add_task(c, name="c_o2")
    snapshots.add_task(phi_s, name="phi_s")
    snapshots.add_task(phi_m, name="phi_m")
    snapshots.add_task(eta, name="eta")
    snapshots.add_task(j_orr, name="j_orr")
    snapshots.add_task(chi_cl, name="chi_cl")

    scalars = solver.evaluator.add_file_handler(
        str(output_dir / "scalars"),
        sim_dt=max(stop_time / 50.0, 1e-7),
        max_writes=100,
    )
    scalars.add_task(d3.Average(c), name="mean_c_o2")
    scalars.add_task(d3.Average(j_orr), name="mean_j_orr_vol")
    scalars.add_task(d3.Integrate(j_orr), name="total_reaction_current")
    scalars.add_task(d3.Average(eta), name="mean_eta")

    return solver


def run(
    *,
    nx: int = 16,
    ny: int = 16,
    nz: int = 48,
    stop_time: float = 2.0e-3,
    max_dt: float = 2.0e-6,
    output_dir: str | Path = "output-electrochem",
) -> None:
    params = CathodeParameters()
    solver = build_solver(
        params,
        nx=nx,
        ny=ny,
        nz=nz,
        stop_time=stop_time,
        output_dir=Path(output_dir),
    )

    logger.info("Starting V0.2 3D open-cathode electrochemistry model")
    logger.info("grid=%dx%dx%d", nx, ny, nz)
    logger.info("open cathode: c_O2,air = %.6g mol/m^3", params.oxygen_inlet_concentration)
    logger.info(
        "temperatures: inlet air=%.2f C, stack/MEA=%.2f C",
        params.oxidant_inlet_temperature - 273.15,
        params.stack_temperature - 273.15,
    )
    logger.info(
        "target stack: %d cells, %.0f W nominal, I=%.2f A, Topt=%.2f C",
        params.stack.n_cells,
        params.stack.rated_power_w,
        params.stack_current_a,
        params.target_stack_temperature_c,
    )
    logger.info(
        "manual-derived air target floor=%.1f slpm, purge period=%.1f s, purge volume=%.0f mL",
        params.target_air_flow_slpm,
        params.purge_period_s,
        params.purge_volume_m3 * 1e6,
    )
    logger.info(
        "electrical BCs: phi_s(air/GDL)=%.3f V, phi_m(membrane)=%.3f V",
        params.cathode_solid_potential,
        params.membrane_proton_potential,
    )

    try:
        while solver.proceed:
            solver.step(max_dt)
            if solver.iteration % 100 == 0:
                logger.info("iteration=%d, t=%.6e", solver.iteration, solver.sim_time)
    except Exception:
        logger.exception("Simulation failed")
        raise
    finally:
        # Dedalus' log_stats() expects the warmup timing markers to exist.
        # Very short smoke tests (<~10 iterations) can finish before those
        # markers are created, so avoid turning a successful smoke run into an
        # AttributeError.
        if hasattr(solver, "warmup_time_end") and hasattr(solver, "warmup_time_start"):
            solver.log_stats()
        else:
            logger.info(
                "Simulation finished before Dedalus warmup statistics were available "
                "(iterations=%d, t=%.6e)",
                solver.iteration,
                solver.sim_time,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--stop-time", type=float, default=2.0e-3)
    parser.add_argument("--max-dt", type=float, default=2.0e-6)
    parser.add_argument("--output-dir", default="output-electrochem")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    run(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        stop_time=args.stop_time,
        max_dt=args.max_dt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
