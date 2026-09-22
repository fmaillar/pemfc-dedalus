"""1D hydrated PEM validation model for V0.3.

This isolates membrane water transport before coupling it to the 3D cathode.
The coordinate z runs from the dry-H2 anode side (z=0) to the humid cathode
side (z=L). Back diffusion is represented by lambda diffusion and
proton-current-driven electro-osmotic drag by an advective lambda flux.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import dedalus.public as d3
import numpy as np

from .membrane import (
    electro_osmotic_lambda_velocity,
    membrane_fixed_charge_concentration,
    membrane_proton_conductivity,
    membrane_water_content_from_activity,
)
from .parameters import CathodeParameters

logger = logging.getLogger(__name__)


def build_solver(
    params: CathodeParameters,
    *,
    nz: int,
    stop_time: float,
    output_dir: Path,
):
    coords = d3.CartesianCoordinates("z")
    dist = d3.Distributor(coords, dtype=np.float64)
    zbasis = d3.ChebyshevT(
        coords["z"],
        size=nz,
        bounds=(0.0, params.membrane_thickness),
        dealias=3 / 2,
    )

    lam = dist.Field(name="lambda", bases=zbasis)
    tau1 = dist.Field(name="tau_lambda1")
    tau2 = dist.Field(name="tau_lambda2")

    lift_basis = zbasis.derivative_basis(1)

    def lift(field, n):
        return d3.Lift(field, lift_basis, n)

    dz = lambda field: d3.Differentiate(field, coords["z"])
    grad_lam = dz(lam) + lift(tau1, -1)

    lambda_anode = float(
        membrane_water_content_from_activity(params.anode_relative_humidity)
    )
    lambda_cathode = float(
        membrane_water_content_from_activity(params.relative_humidity)
    )
    fixed_charge = membrane_fixed_charge_concentration(
        params.membrane_dry_density,
        params.membrane_equivalent_weight,
    )
    drag_velocity = electro_osmotic_lambda_velocity(
        params.membrane_current_density,
        params.faraday,
        fixed_charge,
    )
    diffusivity = params.membrane_water_diffusivity
    length = params.membrane_thickness

    z = dist.local_grid(zbasis)
    lam["g"] = lambda_anode + (lambda_cathode - lambda_anode) * z / length

    problem = d3.IVP([lam, tau1, tau2], namespace=locals())
    problem.add_equation(
        "dt(lam) - diffusivity*dz(grad_lam) + drag_velocity*grad_lam "
        "+ lift(tau2, -1) = 0"
    )
    problem.add_equation("lam(z=0) = lambda_anode")
    problem.add_equation("lam(z=length) = lambda_cathode")

    solver = problem.build_solver(d3.SBDF2)
    solver.stop_sim_time = stop_time
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma_m = membrane_proton_conductivity(lam, params.stack_temperature)
    n_drag = lam / 22.0

    snapshots = solver.evaluator.add_file_handler(
        str(output_dir / "snapshots"),
        sim_dt=max(stop_time / 10.0, 1e-5),
        max_writes=20,
    )
    snapshots.add_task(lam, name="lambda")
    snapshots.add_task(sigma_m, name="sigma_m")
    snapshots.add_task(n_drag, name="n_drag")

    scalars = solver.evaluator.add_file_handler(
        str(output_dir / "scalars"),
        sim_dt=max(stop_time / 50.0, 1e-5),
        max_writes=100,
    )
    scalars.add_task(d3.Average(lam), name="mean_lambda")
    scalars.add_task(d3.Average(sigma_m), name="mean_sigma_m")

    return solver, lambda_anode, lambda_cathode, drag_velocity


def run(
    *,
    nz: int = 64,
    stop_time: float = 0.2,
    max_dt: float = 1e-4,
    output_dir: str | Path = "output-membrane",
) -> None:
    params = CathodeParameters()
    solver, lambda_anode, lambda_cathode, drag_velocity = build_solver(
        params,
        nz=nz,
        stop_time=stop_time,
        output_dir=Path(output_dir),
    )

    logger.info("Starting V0.3 hydrated-membrane validation model")
    logger.info("grid=%d, membrane thickness=%.1f um", nz, params.membrane_thickness * 1e6)
    logger.info(
        "lambda BCs: anode=%.4f, cathode=%.4f; drag velocity=%.3e m/s",
        lambda_anode,
        lambda_cathode,
        drag_velocity,
    )

    try:
        while solver.proceed:
            solver.step(max_dt)
            if solver.iteration % 500 == 0:
                logger.info("iteration=%d, t=%.6e", solver.iteration, solver.sim_time)
    except Exception:
        logger.exception("Simulation failed")
        raise
    finally:
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
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--stop-time", type=float, default=0.2)
    parser.add_argument("--max-dt", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="output-membrane")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    run(
        nz=args.nz,
        stop_time=args.stop_time,
        max_dt=args.max_dt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
