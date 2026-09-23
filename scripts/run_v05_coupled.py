"""V0.5 fixed-point coupling between the 3D cathode and 1D membrane water model.

Each coupling iteration:
1. solves the V0.4 cathode for a membrane-interface proton potential,
2. obtains representative current density from the integrated cathode current,
3. computes steady membrane water transport with diffusion + electro-osmotic drag,
4. integrates membrane protonic area-specific resistance,
5. updates the cathode membrane-interface proton potential.

This is a quasi-steady partitioned coupling. It is deliberately simpler than a
fully monolithic cathode/membrane water PDE and is intended to validate coupling
physics before that larger step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.membrane import (
    electro_osmotic_lambda_velocity,
    membrane_area_specific_resistance,
    membrane_fixed_charge_concentration,
    membrane_proton_conductivity,
    membrane_water_content_from_activity,
    steady_membrane_water_profile,
)
from pemfc_dedalus.parameters import CathodeParameters


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--membrane-nz", type=int, default=129)
    parser.add_argument("--stop-time", type=float, default=0.0064)
    parser.add_argument("--max-dt", type=float, default=1.0e-6)
    parser.add_argument("--scalar-dt", type=float, default=1.0e-5)
    parser.add_argument("--relative-humidity", type=float, default=0.50)
    parser.add_argument("--cathode-solid-potential", type=float, default=0.768)
    parser.add_argument("--max-coupling-iterations", type=int, default=8)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--current-rtol", type=float, default=5.0e-3)
    parser.add_argument("--potential-atol", type=float, default=5.0e-4)
    parser.add_argument("--mpi-n", type=int, default=int(os.environ.get("MPI_N", "8")))
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument(
        "--mpi-flags",
        default=os.environ.get("MPI_FLAGS", "--use-hwthread-cpus"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".v05-coupling"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v05-coupled-membrane.json"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.relative_humidity <= 1.0:
        parser.error("--relative-humidity must be in [0, 1]")
    if not 0.0 < args.relaxation <= 1.0:
        parser.error("--relaxation must be in (0, 1]")
    if args.max_coupling_iterations < 1:
        parser.error("--max-coupling-iterations must be >= 1")

    p = CathodeParameters(
        relative_humidity=args.relative_humidity,
        cathode_solid_potential=args.cathode_solid_potential,
    )
    area_m2 = p.length_x * p.length_y
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    lambda_anode = membrane_water_content_from_activity(
        p.anode_relative_humidity
    ).item()
    lambda_cathode = membrane_water_content_from_activity(
        p.relative_humidity
    ).item()
    z_membrane = np.linspace(0.0, p.membrane_thickness, args.membrane_nz)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    mpi_flags = args.mpi_flags.split()

    phi_m_bc = 0.0
    previous_current: float | None = None
    history: list[dict[str, Any]] = []
    converged = False

    for iteration in range(args.max_coupling_iterations):
        root = args.work_dir / f"iter-{iteration:02d}"
        report_path = root / "report.json"
        root.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                args.mpiexec,
                *mpi_flags,
                "-n",
                str(args.mpi_n),
                "pemfc-cathode-hydrated-3d",
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--nz",
                str(args.nz),
                "--stop-time",
                str(args.stop_time),
                "--max-dt",
                str(args.max_dt),
                "--scalar-dt",
                str(args.scalar_dt),
                "--relative-humidity",
                str(args.relative_humidity),
                "--cathode-solid-potential",
                str(args.cathode_solid_potential),
                "--membrane-proton-potential",
                str(phi_m_bc),
                "--output-dir",
                str(root),
            ],
            env,
        )
        run_command(
            [
                sys.executable,
                "scripts/validate_results.py",
                "--model",
                "v04",
                "--input",
                str(root),
                "--output",
                str(report_path),
                "--relative-humidity",
                str(args.relative_humidity),
            ],
            env,
        )

        report = json.loads(report_path.read_text())
        current_a = float(
            report["scalar_series"]["total_reaction_current"]["last"]
        )
        current_density = current_a / area_m2
        drag_velocity = electro_osmotic_lambda_velocity(
            current_density,
            p.faraday,
            fixed_charge,
        )
        lambda_profile = steady_membrane_water_profile(
            z_membrane,
            lambda_anode=lambda_anode,
            lambda_cathode=lambda_cathode,
            diffusivity_m2_s=p.membrane_water_diffusivity,
            drag_velocity_m_s=drag_velocity,
        )
        asr = membrane_area_specific_resistance(
            z_membrane,
            lambda_profile,
            temperature_k=p.stack_temperature,
            conductivity_floor_s_m=p.membrane_conductivity_floor,
        )
        target_phi_m_bc = -current_density * asr
        relaxed_phi_m_bc = (
            (1.0 - args.relaxation) * phi_m_bc
            + args.relaxation * target_phi_m_bc
        )

        sigma_profile = np.maximum(
            membrane_proton_conductivity(lambda_profile, p.stack_temperature),
            p.membrane_conductivity_floor,
        )
        current_rel_change = (
            None
            if previous_current is None
            else abs(current_a - previous_current) / max(abs(previous_current), 1.0e-30)
        )
        potential_change = abs(relaxed_phi_m_bc - phi_m_bc)

        row = {
            "iteration": iteration,
            "phi_m_bc_v": phi_m_bc,
            "target_phi_m_bc_v": target_phi_m_bc,
            "relaxed_next_phi_m_bc_v": relaxed_phi_m_bc,
            "potential_change_v": potential_change,
            "total_reaction_current_a": current_a,
            "current_density_a_m2": current_density,
            "current_relative_change": current_rel_change,
            "membrane_drag_velocity_m_s": drag_velocity,
            "membrane_asr_ohm_m2": asr,
            "lambda_anode": lambda_anode,
            "lambda_cathode": lambda_cathode,
            "lambda_mean": float(np.mean(lambda_profile)),
            "lambda_min": float(np.min(lambda_profile)),
            "lambda_max": float(np.max(lambda_profile)),
            "sigma_m_mean_s_m": float(np.mean(sigma_profile)),
            "sigma_m_min_s_m": float(np.min(sigma_profile)),
            "sigma_m_max_s_m": float(np.max(sigma_profile)),
            "validation_pass": bool(report["pass"]),
        }
        history.append(row)

        result = {
            "schema_version": 1,
            "model": "v05",
            "grid": [args.nx, args.ny, args.nz],
            "membrane_nz": args.membrane_nz,
            "mpi_ranks": args.mpi_n,
            "relative_humidity": args.relative_humidity,
            "cathode_solid_potential_v": args.cathode_solid_potential,
            "membrane_conductivity_floor_s_m": p.membrane_conductivity_floor,
            "relaxation": args.relaxation,
            "current_rtol": args.current_rtol,
            "potential_atol_v": args.potential_atol,
            "converged": False,
            "history": history,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

        print(
            "coupling "
            f"iter={iteration} I={current_a:.6g} A "
            f"J={current_density:.6g} A/m^2 "
            f"ASR={asr:.6g} ohm m^2 "
            f"phi_m={phi_m_bc:.6g} -> {relaxed_phi_m_bc:.6g} V",
            flush=True,
        )

        if (
            current_rel_change is not None
            and current_rel_change <= args.current_rtol
            and potential_change <= args.potential_atol
        ):
            converged = True
            break

        previous_current = current_a
        phi_m_bc = relaxed_phi_m_bc

    result["converged"] = converged
    result["coupling_iterations"] = len(history)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"coupling converged: {converged}")
    raise SystemExit(0 if converged else 2)


if __name__ == "__main__":
    main()
