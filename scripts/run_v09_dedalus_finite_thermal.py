"""Validate finite-NTU streamwise cathode heating with Dedalus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.cathode_airflow_1d import (
    solve_streamwise_finite_thermal_profile,
)
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ntu-values",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0],
    )
    parser.add_argument(
        "--currents-a",
        nargs="+",
        type=float,
        default=[7.3, 14.5, 26.04],
    )
    parser.add_argument(
        "--inlet-temperatures-c",
        nargs="+",
        type=float,
        default=[10.0, 20.0, 30.0],
    )
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v09-dedalus-finite-thermal.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v09-dedalus-finite-thermal.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for current_a in args.currents_a:
        vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
        heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
        target_temperature_c = p.tech.optimum_stack_temperature_c(current_a)
        target_temperature_k = 273.15 + target_temperature_c
        floor_flow_slpm = p.stack.coolant_air_target_slpm(current_a)

        for inlet_temperature_c in args.inlet_temperatures_c:
            inlet_temperature_k = 273.15 + inlet_temperature_c
            airflow = open_cathode_airflow_target(
                heat_rejection_w=heat_rejection_w,
                inlet_temperature_k=inlet_temperature_k,
                target_stack_temperature_k=target_temperature_k,
                stoichiometric_floor_slpm=floor_flow_slpm,
            )

            if airflow.target_air_flow_slpm is None:
                for ntu in args.ntu_values:
                    summaries.append(
                        {
                            "current_a": current_a,
                            "inlet_temperature_c": inlet_temperature_c,
                            "ntu": ntu,
                            "status": airflow.status,
                        }
                    )
                continue

            for ntu in args.ntu_values:
                solved = solve_streamwise_finite_thermal_profile(
                    current_a=current_a,
                    total_air_flow_slpm=airflow.target_air_flow_slpm,
                    n_cells=p.stack.n_cells,
                    oxygen_mole_fraction=p.oxygen_mole_fraction,
                    faraday_c_mol=p.faraday,
                    inlet_temperature_k=inlet_temperature_k,
                    stack_temperature_k=target_temperature_k,
                    ntu=ntu,
                    points=args.points,
                )
                xi = solved.streamwise_fraction
                expected_temperature = target_temperature_k - (
                    target_temperature_k - inlet_temperature_k
                ) * np.exp(-ntu * xi)
                temperature_error = np.abs(
                    solved.air_temperature_k - expected_temperature
                )
                exact_outlet_temperature_k = target_temperature_k - (
                    target_temperature_k - inlet_temperature_k
                ) * np.exp(-ntu)
                effectiveness = 1.0 - np.exp(-ntu)

                summaries.append(
                    {
                        "current_a": current_a,
                        "inlet_temperature_c": inlet_temperature_c,
                        "ntu": ntu,
                        "status": airflow.status,
                        "active_constraint": airflow.active_constraint,
                        "max_abs_temperature_error_k": float(
                            np.max(temperature_error)
                        ),
                        "outlet_temperature_c": float(
                            exact_outlet_temperature_k - 273.15
                        ),
                        "effectiveness": float(effectiveness),
                        "outlet_approach_to_stack_k": float(
                            target_temperature_k - exact_outlet_temperature_k
                        ),
                    }
                )

                for index, value in enumerate(xi):
                    rows.append(
                        {
                            "current_a": current_a,
                            "inlet_temperature_c": inlet_temperature_c,
                            "ntu": ntu,
                            "streamwise_fraction": float(value),
                            "air_temperature_c": float(
                                solved.air_temperature_k[index] - 273.15
                            ),
                            "analytic_temperature_c": float(
                                expected_temperature[index] - 273.15
                            ),
                            "temperature_error_k": float(
                                temperature_error[index]
                            ),
                            "oxygen_mole_fraction": float(
                                solved.oxygen_mole_fraction[index]
                            ),
                        }
                    )

    reachable = [
        row for row in summaries if "max_abs_temperature_error_k" in row
    ]
    max_temperature_error = max(
        float(row["max_abs_temperature_error_k"])
        for row in reachable
    )

    output = {
        "schema_version": 1,
        "model": "v09-dedalus-finite-thermal-transfer",
        "ntu_values": args.ntu_values,
        "points": args.points,
        "max_abs_temperature_error_k": max_temperature_error,
        "summaries": summaries,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"reachable_rows={len(reachable)} "
        f"max_T_error={max_temperature_error:.3e} K",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
