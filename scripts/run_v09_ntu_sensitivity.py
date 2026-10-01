"""Screen V0.9 sensitivity to finite cathode heat-transfer effectiveness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import (
    airflow_multiplier_for_effectiveness,
    finite_transfer_cooling_power_w,
    heat_exchanger_effectiveness_from_ntu,
    open_cathode_airflow_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ntu-values",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0, 5.0],
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v09-ntu-sensitivity.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v09-ntu-sensitivity.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    rows: list[dict[str, Any]] = []

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

            if airflow.target_air_flow_slpm is None:
                for ntu in args.ntu_values:
                    rows.append(
                        {
                            "current_a": current_a,
                            "inlet_temperature_c": inlet_temperature_c,
                            "ntu": ntu,
                            "status": airflow.status,
                            "active_constraint": airflow.active_constraint,
                            "effectiveness": None,
                            "target_air_flow_slpm": None,
                            "finite_transfer_cooling_w": None,
                            "heat_rejection_w": heat_rejection_w,
                            "cooling_fraction": None,
                            "cooling_deficit_w": None,
                            "ideal_flow_multiplier": None,
                        }
                    )
                continue

            for ntu in args.ntu_values:
                effectiveness = heat_exchanger_effectiveness_from_ntu(ntu)
                cooling_w = finite_transfer_cooling_power_w(
                    stack_temperature_k=273.15 + target_temperature_c,
                    inlet_temperature_k=273.15 + inlet_temperature_c,
                    air_flow_slpm=airflow.target_air_flow_slpm,
                    ntu=ntu,
                )
                cooling_fraction = (
                    cooling_w / heat_rejection_w
                    if heat_rejection_w > 0.0
                    else 1.0
                )
                rows.append(
                    {
                        "current_a": current_a,
                        "inlet_temperature_c": inlet_temperature_c,
                        "ntu": ntu,
                        "status": airflow.status,
                        "active_constraint": airflow.active_constraint,
                        "effectiveness": effectiveness,
                        "target_air_flow_slpm": airflow.target_air_flow_slpm,
                        "finite_transfer_cooling_w": cooling_w,
                        "heat_rejection_w": heat_rejection_w,
                        "cooling_fraction": cooling_fraction,
                        "cooling_deficit_w": max(
                            heat_rejection_w - cooling_w,
                            0.0,
                        ),
                        "ideal_flow_multiplier": (
                            airflow_multiplier_for_effectiveness(effectiveness)
                        ),
                    }
                )

    reachable = [row for row in rows if row["cooling_fraction"] is not None]
    max_deficit_fraction = max(
        max(1.0 - float(row["cooling_fraction"]), 0.0)
        for row in reachable
    )
    max_flow_multiplier = max(
        float(row["ideal_flow_multiplier"])
        for row in reachable
    )

    output = {
        "schema_version": 1,
        "model": "v09-finite-heat-transfer-ntu-sensitivity",
        "ntu_values": args.ntu_values,
        "max_cooling_deficit_fraction": max_deficit_fraction,
        "max_ideal_flow_multiplier": max_flow_multiplier,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"reachable_rows={len(reachable)} "
        f"max_cooling_deficit={100.0 * max_deficit_fraction:.2f}% "
        f"max_flow_multiplier={max_flow_multiplier:.3f}",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
