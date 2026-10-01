"""Screen V0.9 streamwise cathode air heating."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.cathode_airflow import streamwise_air_temperature_profile
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v09-cathode-thermal-profile.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v09-cathode-thermal-profile.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    summaries: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for current_a in args.currents_a:
        vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
        heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
        target_temperature_c = p.tech.optimum_stack_temperature_c(current_a)
        floor_flow_slpm = p.stack.coolant_air_target_slpm(current_a)

        for inlet_temperature_c in args.inlet_temperatures_c:
            airflow = open_cathode_airflow_target(
                heat_rejection_w=heat_rejection_w,
                inlet_temperature_k=273.15 + inlet_temperature_c,
                target_stack_temperature_k=273.15 + target_temperature_c,
                stoichiometric_floor_slpm=floor_flow_slpm,
            )

            summary: dict[str, Any] = {
                "current_a": current_a,
                "inlet_temperature_c": inlet_temperature_c,
                "target_stack_temperature_c": target_temperature_c,
                "heat_rejection_w": heat_rejection_w,
                "status": airflow.status,
                "active_constraint": airflow.active_constraint,
                "target_air_flow_slpm": airflow.target_air_flow_slpm,
                "stoichiometric_floor_slpm": airflow.stoichiometric_floor_slpm,
                "thermal_required_slpm": airflow.thermal_required_slpm,
            }

            if airflow.target_air_flow_slpm is None:
                summary.update(
                    {
                        "outlet_temperature_c": None,
                        "temperature_rise_k": None,
                        "outlet_minus_target_k": None,
                    }
                )
                summaries.append(summary)
                continue

            profile = streamwise_air_temperature_profile(
                inlet_temperature_k=273.15 + inlet_temperature_c,
                total_air_flow_slpm=airflow.target_air_flow_slpm,
                n_cells=p.stack.n_cells,
                total_heat_rejection_w=heat_rejection_w,
                points=args.points,
            )
            outlet_temperature_c = profile.outlet_temperature_k - 273.15
            summary.update(
                {
                    "outlet_temperature_c": outlet_temperature_c,
                    "temperature_rise_k": profile.temperature_rise_k,
                    "outlet_minus_target_k": (
                        outlet_temperature_c - target_temperature_c
                    ),
                }
            )
            summaries.append(summary)

            for index, xi in enumerate(profile.streamwise_fraction):
                profile_rows.append(
                    {
                        "current_a": current_a,
                        "inlet_temperature_c": inlet_temperature_c,
                        "streamwise_fraction": float(xi),
                        "air_temperature_c": float(
                            profile.air_temperature_k[index] - 273.15
                        ),
                    }
                )

    reachable = [
        row for row in summaries if row["temperature_rise_k"] is not None
    ]
    max_rise = max(float(row["temperature_rise_k"]) for row in reachable)

    output = {
        "schema_version": 1,
        "model": "v09-cathode-streamwise-thermal-screening",
        "points": args.points,
        "max_temperature_rise_k": max_rise,
        "summaries": summaries,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(profile_rows[0]),
        )
        writer.writeheader()
        writer.writerows(profile_rows)

    print(
        f"reachable_cases={len(reachable)} "
        f"max_air_temperature_rise={max_rise:.3f} K",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
