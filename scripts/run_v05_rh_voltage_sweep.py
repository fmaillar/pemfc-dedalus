"""Restartable V0.5 RH-voltage sweep using the coupled cathode-membrane model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def case_name(rh: float, voltage: float) -> str:
    return f"rh{int(round(rh * 100)):02d}_v{voltage:.3f}".replace(".", "p")


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rh",
        "cathode_solid_potential_v",
        "converged",
        "coupling_iterations",
        "final_current_a",
        "final_current_density_a_m2",
        "final_phi_m_bc_v",
        "final_target_phi_m_bc_v",
        "final_membrane_asr_ohm_m2",
        "final_lambda_anode",
        "final_lambda_mean",
        "final_lambda_cathode",
        "final_sigma_m_mean_s_m",
        "final_sigma_m_min_s_m",
        "final_sigma_m_max_s_m",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rh-values", nargs="+", type=float, default=[0.30, 0.50, 0.90])
    parser.add_argument("--voltages", nargs="+", type=float, default=[0.70, 0.768, 0.83])
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--membrane-nz", type=int, default=129)
    parser.add_argument("--stop-time", type=float, default=0.0064)
    parser.add_argument("--max-dt", type=float, default=1.0e-6)
    parser.add_argument("--scalar-dt", type=float, default=1.0e-5)
    parser.add_argument("--max-coupling-iterations", type=int, default=8)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--current-rtol", type=float, default=5.0e-3)
    parser.add_argument("--potential-atol", type=float, default=5.0e-4)
    parser.add_argument("--mpi-n", type=int, default=int(os.environ.get("MPI_N", "8")))
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument("--mpi-flags", default=os.environ.get("MPI_FLAGS", "--use-hwthread-cpus"))
    parser.add_argument("--work-dir", type=Path, default=Path(".v05-sweep"))
    parser.add_argument("--output-json", type=Path, default=Path("results/v05-rh-voltage.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/v05-rh-voltage.csv"))
    args = parser.parse_args()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    rows: list[dict[str, Any]] = []
    total_cases = len(args.rh_values) * len(args.voltages)
    completed = 0

    for rh in args.rh_values:
        for voltage in args.voltages:
            name = case_name(rh, voltage)
            case_output = args.work_dir / f"{name}.json"

            if case_output.exists():
                print(f"reusing {case_output}", flush=True)
            else:
                run_command(
                    [
                        sys.executable,
                        "scripts/run_v05_coupled.py",
                        "--nx", str(args.nx),
                        "--ny", str(args.ny),
                        "--nz", str(args.nz),
                        "--membrane-nz", str(args.membrane_nz),
                        "--stop-time", str(args.stop_time),
                        "--max-dt", str(args.max_dt),
                        "--scalar-dt", str(args.scalar_dt),
                        "--relative-humidity", str(rh),
                        "--cathode-solid-potential", str(voltage),
                        "--max-coupling-iterations", str(args.max_coupling_iterations),
                        "--relaxation", str(args.relaxation),
                        "--current-rtol", str(args.current_rtol),
                        "--potential-atol", str(args.potential_atol),
                        "--mpi-n", str(args.mpi_n),
                        "--mpiexec", args.mpiexec,
                        f"--mpi-flags={args.mpi_flags}",
                        "--work-dir", str(args.work_dir / name),
                        "--output", str(case_output),
                    ],
                    env,
                )

            data = json.loads(case_output.read_text())
            history = data["history"]
            if not history:
                raise RuntimeError(f"{name}: empty coupling history")
            last = history[-1]
            rows.append(
                {
                    "rh": rh,
                    "cathode_solid_potential_v": voltage,
                    "converged": bool(data["converged"]),
                    "coupling_iterations": int(data["coupling_iterations"]),
                    "final_current_a": float(last["total_reaction_current_a"]),
                    "final_current_density_a_m2": float(last["current_density_a_m2"]),
                    "final_phi_m_bc_v": float(last["phi_m_bc_v"]),
                    "final_target_phi_m_bc_v": float(last["target_phi_m_bc_v"]),
                    "final_membrane_asr_ohm_m2": float(last["membrane_asr_ohm_m2"]),
                    "final_lambda_anode": float(last["lambda_anode"]),
                    "final_lambda_mean": float(last["lambda_mean"]),
                    "final_lambda_cathode": float(last["lambda_cathode"]),
                    "final_sigma_m_mean_s_m": float(last["sigma_m_mean_s_m"]),
                    "final_sigma_m_min_s_m": float(last["sigma_m_min_s_m"]),
                    "final_sigma_m_max_s_m": float(last["sigma_m_max_s_m"]),
                }
            )
            completed += 1

            summary = {
                "schema_version": 1,
                "model": "v05",
                "grid": [args.nx, args.ny, args.nz],
                "membrane_nz": args.membrane_nz,
                "mpi_ranks": args.mpi_n,
                "relative_humidities": args.rh_values,
                "cathode_solid_potentials_v": args.voltages,
                "completed_cases": completed,
                "total_cases": total_cases,
                "cases": rows,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
            write_csv(args.output_csv, rows)
            print(f"completed cases: {completed} / {total_cases}", flush=True)

    if not all(row["converged"] for row in rows):
        raise SystemExit("one or more V0.5 sweep cases did not converge")


if __name__ == "__main__":
    main()
