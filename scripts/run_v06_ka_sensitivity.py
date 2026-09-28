"""Restartable V0.6 sensitivity study for the anode water-transfer coefficient."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_K_VALUES = (0.0, 5.0e-7, 1.0e-6, 2.0e-6, 5.0e-6, 1.0e-5)

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


def case_name(regime: str, k_value: float) -> str:
    encoded = f"{k_value:.1e}".replace(".", "p").replace("+", "").replace("-", "m")
    return f"{regime}_ka{encoded}"


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def load_v05_reference_rows(path: Path) -> dict[tuple[float, float], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (float(row["rh"]), float(row["cathode_solid_potential_v"])): row
        for row in rows
    }


def v05_reference_to_summary(
    regime: str,
    k_value: float,
    row: dict[str, str],
) -> dict[str, Any]:
    return {
        "regime": regime,
        "anode_water_transfer_coefficient_m_s": k_value,
        "source": "v05_zero_transfer_reference",
        "converged": row["converged"] == "True",
        "coupling_iterations": int(row["coupling_iterations"]),
        "relative_humidity": float(row["rh"]),
        "cathode_solid_potential_v": float(row["cathode_solid_potential_v"]),
        "final_current_a": float(row["final_current_a"]),
        "final_current_density_a_m2": float(row["final_current_density_a_m2"]),
        "final_phi_m_bc_v": float(row["final_phi_m_bc_v"]),
        "final_target_phi_m_bc_v": float(row["final_target_phi_m_bc_v"]),
        "final_membrane_asr_ohm_m2": float(row["final_membrane_asr_ohm_m2"]),
        "final_lambda_anode": float(row["final_lambda_anode"]),
        "final_lambda_mean": float(row["final_lambda_mean"]),
        "final_lambda_cathode": float(row["final_lambda_cathode"]),
        "final_sigma_m_mean_s_m": float(row["final_sigma_m_mean_s_m"]),
        "final_sigma_m_min_s_m": float(row["final_sigma_m_min_s_m"]),
        "final_sigma_m_max_s_m": float(row["final_sigma_m_max_s_m"]),
        "final_anode_water_removal_flux_lambda_m_s": 0.0,
        "final_current_relative_change": None,
        "final_potential_change_v": None,
    }


def v06_result_to_summary(
    regime: str,
    k_value: float,
    data: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    history = data["history"]
    if not history:
        raise ValueError(f"{regime}, k_a={k_value:g}: empty coupling history")
    last = history[-1]
    return {
        "regime": regime,
        "anode_water_transfer_coefficient_m_s": k_value,
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
        raise ValueError("cannot write empty V0.6 sensitivity table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_full_reference_configuration(args: argparse.Namespace) -> bool:
    return (
        args.nx == 16
        and args.ny == 16
        and args.nz == 48
        and args.membrane_nz == 129
        and args.stop_time == 0.0064
        and args.max_dt == 1.0e-6
        and args.scalar_dt == 1.0e-5
        and args.anode_relative_humidity == 0.0
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
        "--k-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--membrane-nz", type=int, default=129)
    parser.add_argument("--stop-time", type=float, default=0.0064)
    parser.add_argument("--max-dt", type=float, default=1.0e-6)
    parser.add_argument("--scalar-dt", type=float, default=1.0e-5)
    parser.add_argument("--anode-relative-humidity", type=float, default=0.0)
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
    parser.add_argument("--work-dir", type=Path, default=Path(".v06-ka-sensitivity"))
    parser.add_argument(
        "--v05-reference-csv",
        type=Path,
        default=Path("results/v05-rh-voltage.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v06-ka-sensitivity.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v06-ka-sensitivity.csv"),
    )
    args = parser.parse_args()

    if any(k < 0.0 for k in args.k_values):
        parser.error("--k-values must all be non-negative")
    if not 0.0 <= args.anode_relative_humidity <= 1.0:
        parser.error("--anode-relative-humidity must be in [0, 1]")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    v05_rows = load_v05_reference_rows(args.v05_reference_csv)
    rows: list[dict[str, Any]] = []
    total_cases = len(args.regimes) * len(args.k_values)
    completed = 0

    for regime in args.regimes:
        rh, voltage = REGIMES[regime]
        for k_value in args.k_values:
            if k_value == 0.0:
                key = (rh, voltage)
                if key not in v05_rows:
                    raise KeyError(f"missing V0.5 reference for {regime}: RH={rh}, V={voltage}")
                row = v05_reference_to_summary(regime, k_value, v05_rows[key])
            else:
                name = case_name(regime, k_value)
                case_dir = args.work_dir / name
                case_output = case_dir / "result.json"

                existing_reference = EXISTING_V06_REFERENCE_FILES[regime]
                use_existing_reference = (
                    k_value == 2.0e-6
                    and is_full_reference_configuration(args)
                    and existing_reference.exists()
                )

                if use_existing_reference:
                    print(f"reusing validated reference {existing_reference}", flush=True)
                    data = json.loads(existing_reference.read_text())
                    row = v06_result_to_summary(
                        regime,
                        k_value,
                        data,
                        "existing_v06_reference",
                    )
                else:
                    data: dict[str, Any] | None = None
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
                        work_dir = case_dir / "iterations"
                        while work_dir.exists():
                            attempt += 1
                            work_dir = case_dir / f"iterations-retry-{attempt:02d}"

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
                                str(rh),
                                "--cathode-solid-potential",
                                str(voltage),
                                "--anode-relative-humidity",
                                str(args.anode_relative_humidity),
                                "--anode-water-transfer-coefficient",
                                str(k_value),
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
                                str(work_dir),
                                "--output",
                                str(case_output),
                            ],
                            env,
                        )
                        data = json.loads(case_output.read_text())

                    row = v06_result_to_summary(regime, k_value, data, "computed_v06")

            rows.append(row)
            completed += 1
            summary = {
                "schema_version": 1,
                "model": "v06-ka-sensitivity",
                "grid": [args.nx, args.ny, args.nz],
                "membrane_nz": args.membrane_nz,
                "mpi_ranks": args.mpi_n,
                "anode_relative_humidity": args.anode_relative_humidity,
                "regimes": args.regimes,
                "k_values_m_s": args.k_values,
                "completed_cases": completed,
                "total_cases": total_cases,
                "cases": rows,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
            write_csv(args.output_csv, rows)
            print(f"completed cases: {completed} / {total_cases}", flush=True)

    if not all(bool(row["converged"]) for row in rows):
        raise SystemExit("one or more V0.6 k_a sensitivity cases did not converge")


if __name__ == "__main__":
    main()
