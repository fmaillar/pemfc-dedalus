"""Evaluate the first V0.8 open-cathode thermal balance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import (
    outlet_temperature_k,
    required_air_flow_slpm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v08-open-cathode-thermal-balance.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v08-open-cathode-thermal-balance.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    current_a = p.stack_current_a
    vcell_v = p.cathode_solid_potential
    heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
    target_temperature_k = 273.15 + p.target_stack_temperature_c
    target_air_flow_slpm = p.target_air_flow_slpm

    ideal_outlet_temperature_k = outlet_temperature_k(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=p.oxidant_inlet_temperature,
        air_flow_slpm=target_air_flow_slpm,
    )
    required_flow_slpm = required_air_flow_slpm(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=p.oxidant_inlet_temperature,
        maximum_outlet_temperature_k=target_temperature_k,
    )

    row = {
        "current_a": current_a,
        "vcell_v": vcell_v,
        "heat_rejection_w": heat_rejection_w,
        "oxidant_inlet_temperature_k": p.oxidant_inlet_temperature,
        "target_stack_temperature_k": target_temperature_k,
        "manual_floor_air_flow_slpm": target_air_flow_slpm,
        "ideal_outlet_temperature_k_at_floor": (
            ideal_outlet_temperature_k
        ),
        "required_air_flow_slpm_for_target_outlet": required_flow_slpm,
        "required_to_floor_flow_ratio": (
            required_flow_slpm / target_air_flow_slpm
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(row, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(
        f"Q={heat_rejection_w:.2f} W, "
        f"air floor={target_air_flow_slpm:.1f} slpm, "
        f"ideal Tout={ideal_outlet_temperature_k - 273.15:.2f} degC, "
        f"target={target_temperature_k - 273.15:.2f} degC, "
        f"required={required_flow_slpm:.1f} slpm",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
