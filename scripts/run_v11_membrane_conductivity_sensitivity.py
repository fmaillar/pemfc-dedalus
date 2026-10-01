"""Sensitivity of V11 polarization to membrane proton conductivity.

The multiplier is applied to the generic Springer conductivity only.  The sweep
is not a calibration: Ballard BOL voltage is reported solely as an external
validation observable and is never used to select a multiplier.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pemfc_dedalus.ballard_1020acs import Ballard1020ACSTechnologyReference
from pemfc_dedalus.v11_polarization import solve_ballard_airflow_operating_point
from pemfc_dedalus.v11_reference import reference_dynamic_scenario


def _run_case(
    current_a: float,
    conductivity_multiplier: float,
    dt_s: float,
    sample_every_s: float,
    max_cycles: int,
    cycle_tolerance: float,
    temperature_tolerance_k: float,
    max_airflow_iterations: int,
    feasibility_scan_points: int,
) -> dict[str, Any]:
    if conductivity_multiplier <= 0.0:
        raise ValueError("conductivity_multiplier must be positive")

    base_inputs = reference_dynamic_scenario().inputs
    inputs = replace(
        base_inputs,
        membrane_conductivity_multiplier=conductivity_multiplier,
    )
    try:
        operating = solve_ballard_airflow_operating_point(
            current_a=current_a,
            dt_s=dt_s,
            sample_every_s=sample_every_s,
            max_cycles=max_cycles,
            cycle_convergence_tolerance=cycle_tolerance,
            temperature_tolerance_k=temperature_tolerance_k,
            max_airflow_iterations=max_airflow_iterations,
            feasibility_scan_points=feasibility_scan_points,
            inputs=inputs,
        )
        point = operating.polarization
        tech = Ballard1020ACSTechnologyReference()
        manual_voltage = tech.bol_typical_cell_voltage_v(current_a)
        row = asdict(point)
        row.pop("final_state")
        row["membrane_conductivity_multiplier"] = conductivity_multiplier
        row["selected_air_flow_slpm"] = operating.selected_air_flow_slpm
        row["airflow_iterations"] = operating.iterations
        row["thermal_target_bracketed"] = operating.thermal_target_bracketed
        row["manual_bol_cell_voltage_v"] = manual_voltage
        row["model_minus_manual_v"] = (
            point.mean_cell_voltage_v - manual_voltage
        )
        row["status"] = "ok"
        row["error"] = ""
        return row
    except Exception as exc:
        return {
            "current_a": current_a,
            "membrane_conductivity_multiplier": conductivity_multiplier,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--currents-a",
        nargs="+",
        type=float,
        default=[15.0, 26.04],
    )
    parser.add_argument(
        "--conductivity-multipliers",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0],
    )
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--sample-every", type=float, default=1.0)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--cycle-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--temperature-tolerance-k", type=float, default=0.5)
    parser.add_argument("--max-airflow-iterations", type=int, default=4)
    parser.add_argument("--feasibility-scan-points", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/v11-membrane-conductivity-sensitivity-quick.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/v11-membrane-conductivity-sensitivity-quick.json"
        ),
    )
    args = parser.parse_args()

    if any(value <= 0.0 for value in args.currents_a):
        parser.error("--currents-a must contain positive values")
    if any(value <= 0.0 for value in args.conductivity_multipliers):
        parser.error("--conductivity-multipliers must contain positive values")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    cases = [
        (current, multiplier)
        for current in args.currents_a
        for multiplier in args.conductivity_multipliers
    ]
    workers = min(args.jobs, len(cases))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_case,
                current,
                multiplier,
                args.dt,
                args.sample_every,
                args.max_cycles,
                args.cycle_tolerance,
                args.temperature_tolerance_k,
                args.max_airflow_iterations,
                args.feasibility_scan_points,
            )
            for current, multiplier in cases
        ]
        rows = [future.result() for future in futures]

    rows.sort(
        key=lambda row: (
            float(row["current_a"]),
            float(row["membrane_conductivity_multiplier"]),
        )
    )

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
        "model": "v11-membrane-conductivity-sensitivity",
        "interpretation": (
            "sensitivity only; Ballard BOL is validation-only and does not "
            "select the conductivity multiplier"
        ),
        "currents_a": args.currents_a,
        "conductivity_multipliers": args.conductivity_multipliers,
        "dt_s": args.dt,
        "max_cycles": args.max_cycles,
        "cycle_tolerance": args.cycle_tolerance,
        "temperature_tolerance_k": args.temperature_tolerance_k,
        "parallel_workers": workers,
        "points": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    for row in rows:
        if row["status"] == "ok":
            print(
                f"I={float(row['current_a']):.3f} A "
                f"sigma_x={float(row['membrane_conductivity_multiplier']):.2f} "
                f"air={float(row['selected_air_flow_slpm']):.2f} slpm "
                f"lambda_mem={float(row['mean_membrane_water_content']):.3f} "
                f"eta_ohm={float(row['mean_membrane_ohmic_loss_v']):.5f} V "
                f"V={float(row['mean_cell_voltage_v']):.6f} V "
                f"dV_BOL={float(row['model_minus_manual_v']):+.6f} V "
                f"dT={float(row['temperature_error_k']):+.3f} K",
                flush=True,
            )
        else:
            print(
                f"I={float(row['current_a']):.3f} A "
                f"sigma_x={float(row['membrane_conductivity_multiplier']):.2f} "
                f"FAILED: {row['error']}",
                flush=True,
            )

    if all(row["status"] == "failed" for row in rows):
        raise RuntimeError("all membrane-conductivity sensitivity cases failed")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
