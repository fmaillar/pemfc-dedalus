"""Post-process V0.8 airflow demand into a lumped stack-temperature trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.ballard_1020acs import UserStackConfiguration
from pemfc_dedalus.thermal import (
    advance_lumped_stack_temperature_k,
    ideal_air_cooling_power_w,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/quick-v08-thermal-dynamic-coupling.csv"),
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("results/quick-v08-thermal-dynamic-coupling.json"),
    )
    parser.add_argument(
        "--initial-stack-temperature-c",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v08-lumped-thermal-dynamics.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v08-lumped-thermal-dynamics.csv"),
    )
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if len(rows) < 2:
        raise ValueError("at least two dynamic samples are required")

    metadata = json.loads(args.input_json.read_text())
    inlet_temperature_c = float(metadata["inlet_temperature_c"])
    inlet_temperature_k = 273.15 + inlet_temperature_c
    stack = UserStackConfiguration()
    thermal_mass_j_k = stack.thermal_mass_j_k

    stack_temperature_k = 273.15 + args.initial_stack_temperature_c
    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        time_s = float(row["time_s"])

        if index > 0:
            previous = rows[index - 1]
            previous_time_s = float(previous["time_s"])
            dt_s = time_s - previous_time_s
            if dt_s <= 0.0:
                raise ValueError("input time must be strictly increasing")

            stack_temperature_k = advance_lumped_stack_temperature_k(
                stack_temperature_k=stack_temperature_k,
                inlet_temperature_k=inlet_temperature_k,
                heat_rejection_w=float(previous["heat_rejection_w"]),
                air_flow_slpm=float(previous["target_air_flow_slpm"]),
                thermal_mass_j_k=thermal_mass_j_k,
                dt_s=dt_s,
            )

        heat_rejection_w = float(row["heat_rejection_w"])
        air_flow_slpm = float(row["target_air_flow_slpm"])
        cooling_w = ideal_air_cooling_power_w(
            stack_temperature_k=stack_temperature_k,
            inlet_temperature_k=inlet_temperature_k,
            air_flow_slpm=air_flow_slpm,
        )
        net_heat_w = heat_rejection_w - cooling_w

        output_rows.append(
            {
                "time_s": time_s,
                "purge_open": row["purge_open"],
                "cell_current_a": float(row["cell_current_a"]),
                "heat_rejection_w": heat_rejection_w,
                "target_air_flow_slpm": air_flow_slpm,
                "stack_temperature_c": stack_temperature_k - 273.15,
                "ideal_air_cooling_w": cooling_w,
                "net_heat_w": net_heat_w,
            }
        )

    temperatures_c = [
        float(row["stack_temperature_c"]) for row in output_rows
    ]
    final_current_a = float(rows[-1]["cell_current_a"])
    summary = {
        "schema_version": 1,
        "model": "v08-lumped-thermal-dynamics",
        "source_csv": str(args.input_csv),
        "source_json": str(args.input_json),
        "thermal_mass_j_k": thermal_mass_j_k,
        "initial_stack_temperature_c": args.initial_stack_temperature_c,
        "inlet_temperature_c": inlet_temperature_c,
        "final_stack_temperature_c": temperatures_c[-1],
        "min_stack_temperature_c": min(temperatures_c),
        "max_stack_temperature_c": max(temperatures_c),
        "final_current_a": final_current_a,
        "final_target_temperature_c": 26.01 + 0.53 * final_current_a,
        "duration_s": float(rows[-1]["time_s"]),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"Cth={thermal_mass_j_k:.1f} J/K, "
        f"T={summary['initial_stack_temperature_c']:.2f}->"
        f"{summary['final_stack_temperature_c']:.2f} C, "
        f"Tmax={summary['max_stack_temperature_c']:.2f} C, "
        f"Ttarget(final)={summary['final_target_temperature_c']:.2f} C",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
