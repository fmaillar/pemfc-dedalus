"""Evaluate the V0.8 parameter-free open-cathode airflow target law."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v08-open-cathode-airflow-control.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v08-open-cathode-airflow-control.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    rows: list[dict[str, Any]] = []

    for current_a in args.currents_a:
        vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
        heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
        target_temperature_c = p.tech.optimum_stack_temperature_c(current_a)
        target_temperature_k = 273.15 + target_temperature_c
        floor_flow_slpm = p.stack.coolant_air_target_slpm(current_a)

        for inlet_temperature_c in args.inlet_temperatures_c:
            result = open_cathode_airflow_target(
                heat_rejection_w=heat_rejection_w,
                inlet_temperature_k=273.15 + inlet_temperature_c,
                target_stack_temperature_k=target_temperature_k,
                stoichiometric_floor_slpm=floor_flow_slpm,
            )
            rows.append(
                {
                    "current_a": current_a,
                    "vcell_bol_typ_v": vcell_v,
                    "heat_rejection_w": heat_rejection_w,
                    "inlet_temperature_c": inlet_temperature_c,
                    "target_stack_temperature_c": target_temperature_c,
                    "status": result.status,
                    "active_constraint": result.active_constraint,
                    "stoichiometric_floor_slpm": (
                        result.stoichiometric_floor_slpm
                    ),
                    "thermal_required_slpm": result.thermal_required_slpm,
                    "target_air_flow_slpm": result.target_air_flow_slpm,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        target = row["target_air_flow_slpm"]
        if target is None:
            target_text = "unreachable"
        else:
            target_text = f"{target:.1f} slpm"
        print(
            f"I={row['current_a']:.2f} A "
            f"Tin={row['inlet_temperature_c']:.1f} C "
            f"constraint={row['active_constraint']} "
            f"target={target_text}",
            flush=True,
        )

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
