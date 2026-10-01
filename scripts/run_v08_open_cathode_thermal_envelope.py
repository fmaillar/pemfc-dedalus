"""Map the V0.8 stationary thermal operating envelope."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import (
    outlet_temperature_k,
    required_air_flow_slpm,
)


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
        default=Path("results/v08-open-cathode-thermal-envelope.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v08-open-cathode-thermal-envelope.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    rows: list[dict[str, Any]] = []

    for current_a in args.currents_a:
        vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
        heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
        target_temperature_c = p.tech.optimum_stack_temperature_c(current_a)
        target_temperature_k = 273.15 + target_temperature_c
        floor_air_flow_slpm = p.stack.coolant_air_target_slpm(current_a)

        for inlet_temperature_c in args.inlet_temperatures_c:
            inlet_temperature_k = 273.15 + inlet_temperature_c
            ideal_outlet_temperature_k = outlet_temperature_k(
                heat_rejection_w=heat_rejection_w,
                inlet_temperature_k=inlet_temperature_k,
                air_flow_slpm=floor_air_flow_slpm,
            )

            feasible = inlet_temperature_k < target_temperature_k
            if feasible:
                required_flow_slpm = required_air_flow_slpm(
                    heat_rejection_w=heat_rejection_w,
                    inlet_temperature_k=inlet_temperature_k,
                    maximum_outlet_temperature_k=target_temperature_k,
                )
                flow_ratio: float | None = (
                    required_flow_slpm / floor_air_flow_slpm
                )
            else:
                required_flow_slpm = None
                flow_ratio = None

            rows.append(
                {
                    "current_a": current_a,
                    "vcell_bol_typ_v": vcell_v,
                    "heat_rejection_w": heat_rejection_w,
                    "inlet_temperature_c": inlet_temperature_c,
                    "target_stack_temperature_c": target_temperature_c,
                    "thermal_margin_k": (
                        target_temperature_k - inlet_temperature_k
                    ),
                    "manual_floor_air_flow_slpm": floor_air_flow_slpm,
                    "ideal_outlet_temperature_c_at_floor": (
                        ideal_outlet_temperature_k - 273.15
                    ),
                    "target_feasible_with_ambient_air": feasible,
                    "required_air_flow_slpm_for_target_outlet": (
                        required_flow_slpm
                    ),
                    "required_to_floor_flow_ratio": flow_ratio,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        required = row["required_air_flow_slpm_for_target_outlet"]
        ratio = row["required_to_floor_flow_ratio"]
        if required is None:
            required_text = "infeasible"
        else:
            required_text = (
                f"{required:.1f} slpm ({ratio:.2f}x floor)"
            )
        print(
            f"I={row['current_a']:.2f} A "
            f"Tin={row['inlet_temperature_c']:.1f} C "
            f"V={row['vcell_bol_typ_v']:.3f} V "
            f"Q={row['heat_rejection_w']:.1f} W "
            f"Topt={row['target_stack_temperature_c']:.1f} C "
            f"required={required_text}",
            flush=True,
        )

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
