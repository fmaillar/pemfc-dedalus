"""Restartable V0.6 spatial-convergence study with finite anode water transfer.

The default campaign targets the most severe validated operating point
(dry_high_load) at k_a = 2e-6 m/s and RH_anode = 0.  It reuses the existing
16x16x48 V0.6 reference and computes 24x24x72 and 32x32x96.

Additional regimes can be requested explicitly.  The quick mode runs a small
8x8x24 -> 12x12x36 pipeline check without reusing full-grid references.
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

REGIMES: dict[str, tuple[float, float]] = {
    "dry_high_load": (0.30, 0.700),
    "nominal": (0.50, 0.768),
    "wet_low_load": (0.90, 0.830),
}

GRIDS = (
    (16, 16, 48),
    (24, 24, 72),
    (32, 32, 96),
)

QUICK_GRIDS = (
    (8, 8, 24),
    (12, 12, 36),
)

EXISTING_V06_REFERENCE_FILES: dict[str, Path] = {
    "dry_high_load": Path("results/v06-dry-high-load.json"),
    "nominal": Path("results/v06-coupled-membrane.json"),
    "wet_low_load": Path("results/v06-wet-low-load.json"),
}


def case_name(regime: str, grid: tuple[int, int, int]) -> str:
    nx, ny, nz = grid
    return f"{regime}_{nx}x{ny}x{nz}"


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def v06_result_to_summary(data: dict[str, Any], source: str) -> dict[str, Any]:
    history = data["history"]
    if not history:
        raise ValueError("empty V0.6 coupling history")
    last = history[-1]
    return {
        "source": source,
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
        "final_anode_water_removal_flux_lambda_m_s": float(
            last["anode_water_removal_flux_lambda_m_s"]
        ),
        "final_current_relative_change": (
            None
            if last["current_relative_change"] is None
            else float(last["current_relative_change"])
        ),
        "final_potential_change_v": float(last["potential_change_v"]),
    }


def rel_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(new), 1.0e-30)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty V0.6 grid-convergence table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def can_reuse_reference(
    *,
    quick: bool,
    grid: tuple[int, int, int],
    membrane_nz: int,
    stop_time: float,
    max_dt: float,
    scalar_dt: float,
    anode_relative_humidity: float,
    k_value: float,
) -> bool:
    return (
        not quick
        and grid == (16, 16, 48)
        and membrane_nz == 129
        and stop_time == 0.0064
        and max_dt == 1.0e-6
        and scalar_dt == 1.0e-5
        and anode_relative_humidity == 0.0
        and k_value == 2.0e-6
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=list(REGIMES),
        default=["dry_high_load"],
    )
    parser.add_argument(
        "--anode-water-transfer-coefficient",
        type=float,
        default=2.0e-6,
    )
    parser.add_argument("--anode-relative-humidity", type=float, default=0.0)
    parser.add_argument("--membrane-nz", type=int, default=129)
    parser.add_argument("--stop-time", type=float, default=0.0064)
    parser.add_argument("--max-dt", type=float, default=1.0e-6)
    parser.add_argument("--scalar-dt", type=float, default=1.0e-5)
    parser.add_argument("--max-coupling-iterations", type=int, default=12)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--current-rtol", type=float, default=5.0e-3)
    parser.add_argument("--potential-atol", type=float, default=5.0e-4)
    parser.add_argument("--mpi-n", type=int, default=int(os.environ.get("MPI_N", "8")))
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument(
        "--mpi-flags",
        default=os.environ.get("MPI_FLAGS", "--use-hwthread-cpus"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".v06-grid-study"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v06-grid-convergence.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v06-grid-convergence.csv"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a small 8x8x24 -> 12x12x36 pipeline check",
    )
    args = parser.parse_args()

    if args.anode_water_transfer_coefficient < 0.0:
        parser.error("--anode-water-transfer-coefficient must be non-negative")
    if not 0.0 <= args.anode_relative_humidity <= 1.0:
        parser.error("--anode-relative-humidity must be in [0, 1]")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    grids = QUICK_GRIDS if args.quick else GRIDS
    rows: list[dict[str, Any]] = []
    point_summaries: list[dict[str, Any]] = []

    for regime in args.regimes:
        rh, voltage = REGIMES[regime]
        levels: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None

        for grid in grids:
            nx, ny, nz = grid
            name = case_name(regime, grid)
            reference = EXISTING_V06_REFERENCE_FILES[regime]

            reuse_reference = (
                can_reuse_reference(
                    quick=args.quick,
                    grid=grid,
                    membrane_nz=args.membrane_nz,
                    stop_time=args.stop_time,
                    max_dt=args.max_dt,
                    scalar_dt=args.scalar_dt,
                    anode_relative_humidity=args.anode_relative_humidity,
                    k_value=args.anode_water_transfer_coefficient,
                )
                and reference.exists()
            )

            data: dict[str, Any] | None
            if reuse_reference:
                print(f"reusing validated reference {reference}", flush=True)
                data = json.loads(reference.read_text())
                summary = v06_result_to_summary(data, "existing_v06_reference")
            else:
                case_dir = args.work_dir / name
                case_output = case_dir / "result.json"
                data = None

                if case_output.exists():
                    candidate = json.loads(case_output.read_text())
                    if bool(candidate.get("converged")):
                        print(f"reusing converged {case_output}", flush=True)
                        data = candidate
                    else:
                        print(
                            f"{case_output} exists but is unconverged; retrying",
                            flush=True,
                        )

                if data is None:
                    attempt = 0
                    iterations_dir = case_dir / "iterations"
                    while iterations_dir.exists():
                        attempt += 1
                        iterations_dir = case_dir / f"iterations-retry-{attempt:02d}"

                    run_command(
                        [
                            sys.executable,
                            "scripts/run_v06_coupled.py",
                            "--nx",
                            str(nx),
                            "--ny",
                            str(ny),
                            "--nz",
                            str(nz),
                            "--membrane-nz",
                            str(args.membrane_nz),
                            "--stop-time",
                            str(args.stop_time),
                            "--max-dt",
                            str(args.max_dt),
                            "--scalar-dt",
                            str(args.scalar_dt),
                            "--relative-humidity",
                            str(rh),
                            "--cathode-solid-potential",
                            str(voltage),
                            "--anode-relative-humidity",
                            str(args.anode_relative_humidity),
                            "--anode-water-transfer-coefficient",
                            str(args.anode_water_transfer_coefficient),
                            "--max-coupling-iterations",
                            str(args.max_coupling_iterations),
                            "--relaxation",
                            str(args.relaxation),
                            "--current-rtol",
                            str(args.current_rtol),
                            "--potential-atol",
                            str(args.potential_atol),
                            "--mpi-n",
                            str(args.mpi_n),
                            "--mpiexec",
                            args.mpiexec,
                            f"--mpi-flags={args.mpi_flags}",
                            "--work-dir",
                            str(iterations_dir),
                            "--output",
                            str(case_output),
                        ],
                        env,
                    )
                    data = json.loads(case_output.read_text())

                summary = v06_result_to_summary(data, "computed_v06")

            row: dict[str, Any] = {
                "regime": regime,
                "relative_humidity": rh,
                "cathode_solid_potential_v": voltage,
                "anode_relative_humidity": args.anode_relative_humidity,
                "anode_water_transfer_coefficient_m_s": (
                    args.anode_water_transfer_coefficient
                ),
                "nx": nx,
                "ny": ny,
                "nz": nz,
                **summary,
                "current_change_from_previous": None,
                "phi_m_change_from_previous": None,
                "asr_change_from_previous": None,
                "lambda_anode_change_from_previous": None,
                "lambda_mean_change_from_previous": None,
                "sigma_min_change_from_previous": None,
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
                row["lambda_anode_change_from_previous"] = rel_change(
                    float(row["final_lambda_anode"]),
                    float(previous["final_lambda_anode"]),
                )
                row["lambda_mean_change_from_previous"] = rel_change(
                    float(row["final_lambda_mean"]),
                    float(previous["final_lambda_mean"]),
                )
                row["sigma_min_change_from_previous"] = rel_change(
                    float(row["final_sigma_m_min_s_m"]),
                    float(previous["final_sigma_m_min_s_m"]),
                )

            rows.append(row)
            levels.append(row)
            previous = row

            output = {
                "schema_version": 1,
                "model": "v06-grid-convergence",
                "quick": args.quick,
                "mpi_ranks": args.mpi_n,
                "membrane_nz": args.membrane_nz,
                "anode_relative_humidity": args.anode_relative_humidity,
                "anode_water_transfer_coefficient_m_s": (
                    args.anode_water_transfer_coefficient
                ),
                "regimes": args.regimes,
                "points": point_summaries
                + [
                    {
                        "regime": regime,
                        "relative_humidity": rh,
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
                "regime": regime,
                "relative_humidity": rh,
                "cathode_solid_potential_v": voltage,
                "levels": levels,
            }
        )

    if not all(bool(row["converged"]) for row in rows):
        raise SystemExit("one or more V0.6 grid-convergence cases did not converge")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
