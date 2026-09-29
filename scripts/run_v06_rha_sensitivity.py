"""Restartable V0.6 sensitivity study for anode relative humidity.

The campaign keeps the finite anode water-transfer coefficient fixed and varies
RH_anode across three representative cathode operating regimes.  The RH_anode=0
points reuse the already validated V0.6 finite-transfer reference results when
the full reference configuration is used.
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

DEFAULT_RH_ANODE_VALUES = (0.0, 0.10, 0.30, 0.50, 0.70, 0.90)

REGIMES: dict[str, tuple[float, float]] = {
    "dry_high_load": (0.30, 0.700),
    "nominal": (0.50, 0.768),
    "wet_low_load": (0.90, 0.830),
}

EXISTING_V06_REFERENCE_FILES: dict[str, Path] = {
    "dry_high_load": Path("results/v06-dry-high-load.json"),
    "nominal": Path("results/v06-coupled-membrane.json"),
    "wet_low_load": Path("results/v06-wet-low-load.json"),
}


def case_name(regime: str, rh_anode: float) -> str:
    encoded = f"{rh_anode:.2f}".replace(".", "p")
    return f"{regime}_rha{encoded}"


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def v06_result_to_summary(
    regime: str,
    rh_anode: float,
    data: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    history = data["history"]
    if not history:
        raise ValueError(f"{regime}, RH_anode={rh_anode:g}: empty coupling history")
    last = history[-1]
    return {
        "regime": regime,
        "anode_relative_humidity": rh_anode,
        "lambda_anode_equilibrium": float(data["lambda_anode_equilibrium"]),
        "anode_water_transfer_coefficient_m_s": float(
            data["anode_water_transfer_coefficient_m_s"]
        ),
        "source": source,
        "converged": bool(data["converged"]),
        "coupling_iterations": int(data["coupling_iterations"]),
        "relative_humidity": float(data["relative_humidity"]),
        "cathode_solid_potential_v": float(data["cathode_solid_potential_v"]),
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty V0.6 RH_anode sensitivity table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def can_reuse_zero_rh_reference(
    *,
    rh_anode: float,
    k_value: float,
    nx: int,
    ny: int,
    nz: int,
    membrane_nz: int,
    stop_time: float,
    max_dt: float,
    scalar_dt: float,
) -> bool:
    return (
        rh_anode == 0.0
        and k_value == 2.0e-6
        and nx == 16
        and ny == 16
        and nz == 48
        and membrane_nz == 129
        and stop_time == 0.0064
        and max_dt == 1.0e-6
        and scalar_dt == 1.0e-5
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=list(REGIMES),
        default=list(REGIMES),
    )
    parser.add_argument(
        "--anode-rh-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_RH_ANODE_VALUES),
    )
    parser.add_argument(
        "--anode-water-transfer-coefficient",
        type=float,
        default=2.0e-6,
    )
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=48)
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
    parser.add_argument("--work-dir", type=Path, default=Path(".v06-rha-sensitivity"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v06-rha-sensitivity.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v06-rha-sensitivity.csv"),
    )
    args = parser.parse_args()

    if any(not 0.0 <= rh <= 1.0 for rh in args.anode_rh_values):
        parser.error("--anode-rh-values must all be in [0, 1]")
    if args.anode_water_transfer_coefficient < 0.0:
        parser.error("--anode-water-transfer-coefficient must be non-negative")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    rows: list[dict[str, Any]] = []
    total_cases = len(args.regimes) * len(args.anode_rh_values)
    completed = 0

    for regime in args.regimes:
        rh_cathode, voltage = REGIMES[regime]

        for rh_anode in args.anode_rh_values:
            case_dir = args.work_dir / case_name(regime, rh_anode)
            case_output = case_dir / "result.json"
            reference = EXISTING_V06_REFERENCE_FILES[regime]

            reuse_reference = (
                can_reuse_zero_rh_reference(
                    rh_anode=rh_anode,
                    k_value=args.anode_water_transfer_coefficient,
                    nx=args.nx,
                    ny=args.ny,
                    nz=args.nz,
                    membrane_nz=args.membrane_nz,
                    stop_time=args.stop_time,
                    max_dt=args.max_dt,
                    scalar_dt=args.scalar_dt,
                )
                and reference.exists()
            )

            data: dict[str, Any] | None
            source: str

            if reuse_reference:
                print(f"reusing validated reference {reference}", flush=True)
                data = json.loads(reference.read_text())
                source = "existing_v06_zero_rh_reference"
            else:
                data = None
                source = "computed_v06"

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
                            str(args.nx),
                            "--ny",
                            str(args.ny),
                            "--nz",
                            str(args.nz),
                            "--membrane-nz",
                            str(args.membrane_nz),
                            "--stop-time",
                            str(args.stop_time),
                            "--max-dt",
                            str(args.max_dt),
                            "--scalar-dt",
                            str(args.scalar_dt),
                            "--relative-humidity",
                            str(rh_cathode),
                            "--cathode-solid-potential",
                            str(voltage),
                            "--anode-relative-humidity",
                            str(rh_anode),
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

            row = v06_result_to_summary(regime, rh_anode, data, source)
            rows.append(row)
            completed += 1

            summary = {
                "schema_version": 1,
                "model": "v06-rha-sensitivity",
                "grid": [args.nx, args.ny, args.nz],
                "membrane_nz": args.membrane_nz,
                "mpi_ranks": args.mpi_n,
                "anode_water_transfer_coefficient_m_s": (
                    args.anode_water_transfer_coefficient
                ),
                "regimes": args.regimes,
                "anode_rh_values": args.anode_rh_values,
                "completed_cases": completed,
                "total_cases": total_cases,
                "cases": rows,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
            write_csv(args.output_csv, rows)
            print(f"completed cases: {completed} / {total_cases}", flush=True)

    if not all(bool(row["converged"]) for row in rows):
        raise SystemExit("one or more V0.6 RH_anode sensitivity cases did not converge")


if __name__ == "__main__":
    main()
