"""Spatial convergence study for V0.5 coupled cathode-membrane cases.

Reuses the completed 16x16x48 V0.5 RH-voltage sweep as the coarse level and
computes 24x24x72 and 32x32x96 for three representative operating points.
The study is restartable and writes consolidated JSON/CSV results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


POINTS = (
    (0.30, 0.700, "dry_high_load"),
    (0.50, 0.768, "nominal"),
    (0.90, 0.830, "wet_low_load"),
)

GRIDS = (
    (16, 16, 48),
    (24, 24, 72),
    (32, 32, 96),
)


def case_name(label: str, grid: tuple[int, int, int]) -> str:
    nx, ny, nz = grid
    return f"{label}_{nx}x{ny}x{nz}"


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def load_baseline(
    path: Path,
    *,
    rh: float,
    voltage: float,
) -> dict[str, Any]:
    data = json.loads(path.read_text())
    for case in data["cases"]:
        if (
            abs(float(case["rh"]) - rh) < 1.0e-12
            and abs(float(case["cathode_solid_potential_v"]) - voltage) < 1.0e-12
        ):
            return {
                "converged": bool(case["converged"]),
                "coupling_iterations": int(case["coupling_iterations"]),
                "final_current_a": float(case["final_current_a"]),
                "final_current_density_a_m2": float(case["final_current_density_a_m2"]),
                "final_phi_m_bc_v": float(case["final_phi_m_bc_v"]),
                "final_target_phi_m_bc_v": float(case["final_target_phi_m_bc_v"]),
                "final_membrane_asr_ohm_m2": float(case["final_membrane_asr_ohm_m2"]),
                "final_lambda_anode": float(case["final_lambda_anode"]),
                "final_lambda_mean": float(case["final_lambda_mean"]),
                "final_lambda_cathode": float(case["final_lambda_cathode"]),
                "final_sigma_m_mean_s_m": float(case["final_sigma_m_mean_s_m"]),
                "final_sigma_m_min_s_m": float(case["final_sigma_m_min_s_m"]),
                "final_sigma_m_max_s_m": float(case["final_sigma_m_max_s_m"]),
            }
    raise KeyError(f"baseline case not found for RH={rh}, V={voltage}")


def summarize_case(data: dict[str, Any]) -> dict[str, Any]:
    history = data["history"]
    if not history:
        raise RuntimeError("empty V0.5 coupling history")
    last = history[-1]
    return {
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


def rel_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(new), 1.0e-30)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "rh",
        "cathode_solid_potential_v",
        "nx",
        "ny",
        "nz",
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
        "source",
        "current_change_from_previous",
        "phi_m_change_from_previous",
        "asr_change_from_previous",
        "lambda_mean_change_from_previous",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/v05-rh-voltage.json"),
    )
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
    parser.add_argument(
        "--mpi-flags",
        default=os.environ.get("MPI_FLAGS", "--use-hwthread-cpus"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".v05-grid-study"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v05-grid-convergence.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v05-grid-convergence.csv"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only the nominal case on 8x8x24 and 12x12x36",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    points = ((0.50, 0.768, "nominal"),) if args.quick else POINTS
    grids = ((8, 8, 24), (12, 12, 36)) if args.quick else GRIDS

    rows: list[dict[str, Any]] = []
    point_summaries: list[dict[str, Any]] = []

    for rh, voltage, label in points:
        levels: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None

        for grid_index, grid in enumerate(grids):
            nx, ny, nz = grid
            name = case_name(label, grid)

            if not args.quick and grid_index == 0:
                summary = load_baseline(args.baseline, rh=rh, voltage=voltage)
                source = str(args.baseline)
            else:
                case_output = args.work_dir / f"{name}.json"
                if case_output.exists():
                    print(f"reusing {case_output}", flush=True)
                else:
                    run_command(
                        [
                            sys.executable,
                            "scripts/run_v05_coupled.py",
                            "--nx", str(nx),
                            "--ny", str(ny),
                            "--nz", str(nz),
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
                summary = summarize_case(data)
                source = str(case_output)

            row: dict[str, Any] = {
                "label": label,
                "rh": rh,
                "cathode_solid_potential_v": voltage,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                **summary,
                "source": source,
                "current_change_from_previous": None,
                "phi_m_change_from_previous": None,
                "asr_change_from_previous": None,
                "lambda_mean_change_from_previous": None,
            }

            if previous is not None:
                row["current_change_from_previous"] = rel_change(
                    float(row["final_current_a"]),
                    float(previous["final_current_a"]),
                )
                row["phi_m_change_from_previous"] = rel_change(
                    float(row["final_phi_m_bc_v"]),
                    float(previous["final_phi_m_bc_v"]),
                )
                row["asr_change_from_previous"] = rel_change(
                    float(row["final_membrane_asr_ohm_m2"]),
                    float(previous["final_membrane_asr_ohm_m2"]),
                )
                row["lambda_mean_change_from_previous"] = rel_change(
                    float(row["final_lambda_mean"]),
                    float(previous["final_lambda_mean"]),
                )

            rows.append(row)
            levels.append(row)
            previous = row

            output = {
                "schema_version": 1,
                "model": "v05",
                "quick": args.quick,
                "mpi_ranks": args.mpi_n,
                "membrane_nz": args.membrane_nz,
                "points": point_summaries
                + [
                    {
                        "label": label,
                        "rh": rh,
                        "cathode_solid_potential_v": voltage,
                        "levels": levels,
                    }
                ],
                "rows": rows,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(output, indent=2) + "\n")
            write_csv(args.output_csv, rows)

        point_summaries.append(
            {
                "label": label,
                "rh": rh,
                "cathode_solid_potential_v": voltage,
                "levels": levels,
            }
        )

    if not all(bool(row["converged"]) for row in rows):
        raise SystemExit("one or more V0.5 grid-convergence cases did not converge")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
