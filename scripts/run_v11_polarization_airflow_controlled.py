"""Run V11 polarization with one cathode-air stream meeting thermal target."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pemfc_dedalus.ballard_1020acs import Ballard1020ACSTechnologyReference
from pemfc_dedalus.v11_materials import V11MEAReference
from pemfc_dedalus.v11_polarization import solve_ballard_airflow_operating_point


def _run_current(
    current_a: float,
    dt_s: float,
    sample_every_s: float,
    max_cycles: int,
    cycle_tolerance: float,
    temperature_tolerance_k: float,
    max_airflow_iterations: int,
    feasibility_scan_points: int,
) -> dict[str, Any]:
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
        )
        point = operating.polarization
        tech = Ballard1020ACSTechnologyReference()
        manual_voltage = tech.bol_typical_cell_voltage_v(current_a)
        row = asdict(point)
        row.pop("final_state")
        row["selected_air_flow_slpm"] = operating.selected_air_flow_slpm
        row["minimum_air_flow_slpm"] = operating.minimum_air_flow_slpm
        row["airflow_iterations"] = operating.iterations
        row["thermal_target_bracketed"] = operating.thermal_target_bracketed
        row["status"] = "ok"
        row["error"] = ""
        row["manual_bol_cell_voltage_v"] = manual_voltage
        row["model_minus_manual_v"] = (
            point.mean_cell_voltage_v - manual_voltage
        )

        mea = V11MEAReference()
        current_density = mea.current_density_a_m2(current_a)
        implied_ohmic = (
            point.mean_reversible_voltage_v
            - point.mean_activation_loss_v
            - point.mean_additional_resolved_loss_v
            - manual_voltage
        )
        row["bol_implied_membrane_ohmic_loss_v"] = implied_ohmic
        row["model_membrane_asr_ohm_m2"] = (
            point.mean_membrane_ohmic_loss_v / current_density
        )
        row["bol_implied_membrane_asr_ohm_m2"] = (
            implied_ohmic / current_density
        )
        row["bol_implied_effective_membrane_thickness_m"] = (
            mea.membrane_thickness_m
            * implied_ohmic
            / point.mean_membrane_ohmic_loss_v
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
    parser.add_argument("--cycle-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--temperature-tolerance-k", type=float, default=0.10)
    parser.add_argument("--max-airflow-iterations", type=int, default=8)
    parser.add_argument("--feasibility-scan-points", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v11-polarization-airflow-controlled.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v11-polarization-airflow-controlled.json"),
    )
    args = parser.parse_args()

    if not args.currents_a or any(value <= 0.0 for value in args.currents_a):
        parser.error("--currents-a must contain positive values")
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
                args.cycle_tolerance,
                args.temperature_tolerance_k,
                args.max_airflow_iterations,
                args.feasibility_scan_points,
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
        raise RuntimeError("all airflow-controlled V11 points failed")

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
        "model": "v11-unified-airflow-controlled-polarization",
        "dt_s": args.dt,
        "sample_every_s": args.sample_every,
        "max_cycles": args.max_cycles,
        "cycle_tolerance": args.cycle_tolerance,
        "temperature_tolerance_k": args.temperature_tolerance_k,
        "max_airflow_iterations": args.max_airflow_iterations,
        "feasibility_scan_points": args.feasibility_scan_points,
        "parallel_workers": workers,
        "points": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    for row in rows:
        if row["status"] == "ok":
            print(
                f"I={float(row['current_a']):.3f} A "
                f"air={float(row['selected_air_flow_slpm']):.3f} slpm "
                f"lambdaO2={float(row['oxygen_stoichiometry']):.2f} "
                f"T={float(row['mean_stack_temperature_k']) - 273.15:.3f} C "
                f"Ttarget={float(row['target_stack_temperature_k']) - 273.15:.3f} C "
                f"dT={float(row['temperature_error_k']):+.3f} K "
                f"Vmean={float(row['mean_cell_voltage_v']):.6f} V "
                f"Erev={float(row['mean_reversible_voltage_v']):.6f} V "
                f"eta_act={float(row['mean_activation_loss_v']):.6f} V "
                f"eta_ohm={float(row['mean_membrane_ohmic_loss_v']):.6f} V "
                f"dVmanual={float(row['model_minus_manual_v']):+.6f} V "
                f"cycles={int(row['cycles_completed'])} "
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
