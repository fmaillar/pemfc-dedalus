"""3D cathode GDL+CL oxygen diffusion-reaction model.

This is the first verification model, not yet a complete PEMFC.

Coordinates:
    x: streamwise, periodic
    y: transverse channel/rib direction, periodic
    z: through-plane, Chebyshev; z=0 gas side, z=Lz membrane side

PDE:
    eps * dt(c) = div(D_eff grad(c)) - k_rxn * chi_CL * c

The top Dirichlet boundary is modulated in y to force a genuinely 3D field.
The membrane-side boundary is zero-flux.
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
        coords["x"],
        size=nx,
        bounds=(0.0, params.length_x),
        dealias=3 / 2,
    )
    ybasis = d3.RealFourier(
        coords["y"],
        size=ny,
        bounds=(0.0, params.length_y),
        dealias=3 / 2,
    )
    zbasis = d3.ChebyshevT(
        coords["z"],
        size=nz,
        bounds=(0.0, params.thickness_z),
        dealias=3 / 2,
    )
    bases = (xbasis, ybasis, zbasis)

    c = dist.Field(name="c_o2", bases=bases)

    tau1 = dist.Field(name="tau1", bases=(xbasis, ybasis))
    tau2 = dist.Field(name="tau2", bases=(xbasis, ybasis))

    x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

    z_interface = params.gdl_thickness
    chi_cl = dist.Field(name="chi_cl", bases=zbasis)
    z1 = dist.local_grid(zbasis)
    chi_cl["g"] = 0.5 * (
        1.0 + np.tanh((z1 - z_interface) / params.interface_width)
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

    c_in = params.oxygen_inlet_concentration
    inlet = dist.Field(name="c_inlet", bases=(xbasis, ybasis))
    x2, y2 = dist.local_grids(xbasis, ybasis)
    inlet["g"] = c_in * (
        0.90
        + 0.10
        * np.cos(2.0 * np.pi * y2 / params.length_y)
        * (0.95 + 0.05 * np.cos(2.0 * np.pi * x2 / params.length_x))
    )

    c["g"] = c_in * (
        1.0
        + 1e-3
        * np.cos(2.0 * np.pi * x / params.length_x)
        * np.cos(2.0 * np.pi * y / params.length_y)
        * np.cos(np.pi * z / params.thickness_z)
    )

    grad = d3.grad
    div = d3.div

    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    ez = coords.unit_vector_fields(dist)[2]

    grad_c = grad(c) + ez * lift(tau1, -1)

    d_ref = min(params.d_o2_gdl, params.d_o2_cl)
    k_reaction = params.k_reaction
    Lz = params.thickness_z

    problem = d3.IVP([c, tau1, tau2], namespace=locals())
    problem.add_equation(
        "porosity*dt(c) - d_ref*div(grad_c) + lift(tau2, -1) "
        "= div((diffusivity-d_ref)*grad_c) - k_reaction*chi_cl*c"
    )
    problem.add_equation("c(z=0) = inlet")
    problem.add_equation("ez @ grad_c(z=Lz) = 0")

    solver = problem.build_solver(d3.SBDF2)
    solver.stop_sim_time = stop_time

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = solver.evaluator.add_file_handler(
        str(output_dir / "snapshots"),
        sim_dt=max(stop_time / 10.0, 1e-6),
        max_writes=20,
    )
    snapshots.add_task(c, name="c_o2")
    snapshots.add_task(chi_cl, name="chi_cl")
    snapshots.add_task(diffusivity, name="d_eff")

    scalars = solver.evaluator.add_file_handler(
        str(output_dir / "scalars"),
        sim_dt=max(stop_time / 50.0, 1e-6),
        max_writes=100,
    )
    scalars.add_task(d3.Average(c), name="mean_c_o2")
    scalars.add_task(
        d3.Integrate(k_reaction * chi_cl * c),
        name="o2_sink_integral",
    )

    return solver


def run(
    *,
    nx: int = 32,
    ny: int = 32,
    nz: int = 64,
    stop_time: float = 0.05,
    max_dt: float = 5e-5,
    output_dir: str | Path = "output",
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

    logger.info("Starting 3D cathode diffusion-reaction model")
    logger.info(
        "grid=%dx%dx%d, L=(%.3g, %.3g, %.3g) m",
        nx,
        ny,
        nz,
        params.length_x,
        params.length_y,
        params.thickness_z,
    )
    logger.info("c_O2,in = %.6g mol/m^3", params.oxygen_inlet_concentration)
    logger.info(
        "D_eff(GDL)=%.6g, D_eff(CL)=%.6g m^2/s",
        params.d_o2_gdl,
        params.d_o2_cl,
    )

    try:
        while solver.proceed:
            solver.step(max_dt)
            if solver.iteration % 100 == 0:
                logger.info(
                    "iteration=%d, t=%.6e",
                    solver.iteration,
                    solver.sim_time,
                )
    except Exception:
        logger.exception("Simulation failed")
        raise
    finally:
        solver.log_stats()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--stop-time", type=float, default=0.05)
    parser.add_argument("--max-dt", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="output")
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
