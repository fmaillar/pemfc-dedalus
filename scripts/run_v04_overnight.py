"""Overnight V0.4 RH x cathode-voltage study on one fixed fine grid.

The campaign is intentionally restartable: completed cases with a valid report.json
are skipped, so an interrupted overnight run can be resumed safely.
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

DEFAULT_RH = (0.10, 0.30, 0.50, 0.70, 0.90)
DEFAULT_VOLTAGES = (0.62, 0.70, 0.768, 0.83)


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def case_name(rh: float, voltage: float) -> str:
    return f"rh{int(round(100 * rh)):02d}_v{voltage:.3f}".replace(".", "p")


def scalar_last(report: dict[str, Any], name: str) -> float:
    return float(report["scalar_series"][name]["last"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--stop-time", type=float, default=0.0064)
    parser.add_argument("--max-dt", type=float, default=1.0e-6)
    parser.add_argument("--scalar-dt", type=float, default=1.0e-5)
    parser.add_argument("--mpi-n", type=int, default=int(os.environ.get("MPI_N", "8")))
    parser.add_argument(
        "--rh-values",
        default=",".join(str(value) for value in DEFAULT_RH),
        help="comma-separated relative humidities as fractions",
    )
    parser.add_argument(
        "--voltages",
        default=",".join(str(value) for value in DEFAULT_VOLTAGES),
        help="comma-separated cathode solid potentials [V]",
    )
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument(
        "--mpi-flags",
        default=os.environ.get("MPI_FLAGS", "--use-hwthread-cpus"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".overnight-output/v04-rh-voltage"))
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/overnight-v04-rh-voltage.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("results/overnight-v04-rh-voltage.csv"),
    )
    args = parser.parse_args()

    if args.mpi_n <= 0:
        parser.error("--mpi-n must be positive")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    rows: list[dict[str, Any]] = []
    mpi_flags = args.mpi_flags.split()
    rh_values = tuple(float(value) for value in args.rh_values.split(","))
    voltages = tuple(float(value) for value in args.voltages.split(","))

    for rh in rh_values:
        if not 0.0 <= rh <= 1.0:
            parser.error("all --rh-values entries must be between 0 and 1")
        for voltage in voltages:
            name = case_name(rh, voltage)
            root = args.work_dir / name
            report_path = root / "report.json"

            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get("pass") is True:
                    print(f"skip completed case {name}", flush=True)
                else:
                    report_path.unlink()
                    report = {}
            else:
                report = {}

            if not report:
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
                        str(rh),
                        "--cathode-solid-potential",
                        str(voltage),
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
                    ],
                    env,
                )
                report = json.loads(report_path.read_text())

            fields = report["fields_last_write"]
            rows.append(
                {
                    "rh": rh,
                    "cathode_solid_potential_v": voltage,
                    "grid": [args.nx, args.ny, args.nz],
                    "mpi_ranks": args.mpi_n,
                    "stop_time_s": args.stop_time,
                    "pass": bool(report["pass"]),
                    "lambda_cl_max": float(fields["lambda_cl"]["max"]),
                    "sigma_m_max_s_per_m": float(fields["sigma_m"]["max"]),
                    "mean_c_o2_mol_per_m3": scalar_last(report, "mean_c_o2"),
                    "mean_eta_v": scalar_last(report, "mean_eta"),
                    "total_reaction_current_a": scalar_last(
                        report, "total_reaction_current"
                    ),
                    "j_orr_max_a_per_m3": float(fields["j_orr"]["max"]),
                    "phi_m_min_v": float(fields["phi_m"]["min"]),
                    "phi_m_max_v": float(fields["phi_m"]["max"]),
                }
            )

            summary = {
                "schema_version": 1,
                "grid": [args.nx, args.ny, args.nz],
                "mpi_ranks": args.mpi_n,
                "stop_time_s": args.stop_time,
                "relative_humidities": list(rh_values),
                "cathode_solid_potentials_v": list(voltages),
                "cases": rows,
            }
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(summary, indent=2) + "\n")

            with args.csv_output.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "rh",
                        "cathode_solid_potential_v",
                        "mpi_ranks",
                        "stop_time_s",
                        "pass",
                        "lambda_cl_max",
                        "sigma_m_max_s_per_m",
                        "mean_c_o2_mol_per_m3",
                        "mean_eta_v",
                        "total_reaction_current_a",
                        "j_orr_max_a_per_m3",
                        "phi_m_min_v",
                        "phi_m_max_v",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    csv_row = dict(row)
                    csv_row.pop("grid")
                    writer.writerow(csv_row)

    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    print(f"completed cases: {len(rows)} / {len(rh_values) * len(voltages)}")


if __name__ == "__main__":
    main()
