"""Run the V11 purge-periodic polarization curve in parallel."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pemfc_dedalus.ballard_1020acs import Ballard1020ACSTechnologyReference
from pemfc_dedalus.v11_polarization import simulate_periodic_polarization_point


def _run_current(
    current_a: float,
    dt_s: float,
    sample_every_s: float,
    max_cycles: int,
    convergence_tolerance: float,
) -> dict[str, Any]:
    try:
        point = simulate_periodic_polarization_point(
            current_a=current_a,
            dt_s=dt_s,
            sample_every_s=sample_every_s,
            max_cycles=max_cycles,
            convergence_tolerance=convergence_tolerance,
        )
        tech = Ballard1020ACSTechnologyReference()
        manual_voltage = tech.bol_typical_cell_voltage_v(current_a)
        row = asdict(point)
        row.pop("final_state")
        row["status"] = "ok"
        row["error"] = ""
        row["manual_bol_cell_voltage_v"] = manual_voltage
        row["model_minus_manual_v"] = (
            point.mean_cell_voltage_v - manual_voltage
        )
        return row
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "current_a": current_a,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--currents-a",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 15.0, 20.0, 26.04, 30.0],
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sample-every", type=float, default=1.0)
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--convergence-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v11-polarization-periodic.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v11-polarization-periodic.json"),
    )
    args = parser.parse_args()

    if not args.currents_a or any(value <= 0.0 for value in args.currents_a):
        parser.error("--currents-a must contain positive values")
    if args.dt <= 0.0 or args.sample_every <= 0.0:
        parser.error("--dt and --sample-every must be positive")
    if args.max_cycles < 2:
        parser.error("--max-cycles must be >= 2")
    if args.convergence_tolerance <= 0.0:
        parser.error("--convergence-tolerance must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    workers = min(args.jobs, len(args.currents_a))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_current,
                current_a,
                args.dt,
                args.sample_every,
                args.max_cycles,
                args.convergence_tolerance,
            )
            for current_a in args.currents_a
        ]
        rows = [future.result() for future in futures]

    rows.sort(key=lambda row: float(row["current_a"]))
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        for row in rows:
            print(
                f"I={float(row['current_a']):.3f} A FAILED: {row['error']}",
                flush=True,
            )
        raise RuntimeError("all V11 polarization current points failed")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output = {
        "schema_version": 1,
        "model": "v11-purge-periodic-polarization",
        "dt_s": args.dt,
        "sample_every_s": args.sample_every,
        "max_cycles": args.max_cycles,
        "convergence_tolerance": args.convergence_tolerance,
        "parallel_workers": workers,
        "note": (
            "Manual BOL voltages are validation-only. V11 cathode Pt loading "
            "and ECSA remain explicit generic sensitivity assumptions."
        ),
        "points": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    for row in rows:
        if row["status"] == "ok":
            print(
                f"I={float(row['current_a']):.3f} A "
                f"Vmean={float(row['mean_cell_voltage_v']):.6f} V "
                f"Vmanual={float(row['manual_bol_cell_voltage_v']):.6f} V "
                f"dV={float(row['model_minus_manual_v']):+.6f} V "
                f"cycles={int(row['cycles_completed'])} "
                f"cycle_err={float(row['cycle_state_error']):.3e} "
                f"converged={bool(row['converged'])}",
                flush=True,
            )
        else:
            print(
                f"I={float(row['current_a']):.3f} A FAILED: {row['error']}",
                flush=True,
            )

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
