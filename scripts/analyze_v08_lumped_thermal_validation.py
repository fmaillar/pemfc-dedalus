"""Validate the V0.8 ideal lumped thermal model against its analytic solution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.ballard_1020acs import UserStackConfiguration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import (
    advance_lumped_stack_temperature_k,
    lumped_temperature_analytic_k,
    lumped_thermal_time_constant_s,
    open_cathode_airflow_target,
)


def integrate_constant_case(
    *,
    initial_temperature_k: float,
    inlet_temperature_k: float,
    heat_rejection_w: float,
    air_flow_slpm: float,
    thermal_mass_j_k: float,
    duration_s: float,
    dt_s: float,
) -> float:
    temperature = initial_temperature_k
    n_steps = int(round(duration_s / dt_s))
    for _ in range(n_steps):
        temperature = advance_lumped_stack_temperature_k(
            stack_temperature_k=temperature,
            inlet_temperature_k=inlet_temperature_k,
            heat_rejection_w=heat_rejection_w,
            air_flow_slpm=air_flow_slpm,
            thermal_mass_j_k=thermal_mass_j_k,
            dt_s=dt_s,
        )
    return temperature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inlet-temperatures-c",
        nargs="+",
        type=float,
        default=[10.0, 20.0, 30.0],
    )
    parser.add_argument(
        "--initial-offsets-k",
        nargs="+",
        type=float,
        default=[0.0, 10.0],
    )
    parser.add_argument(
        "--time-steps-s",
        nargs="+",
        type=float,
        default=[2.0, 1.0, 0.5, 0.25],
    )
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v08-lumped-thermal-validation.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v08-lumped-thermal-validation.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    stack = UserStackConfiguration()
    current_a = p.stack_current_a
    vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
    heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
    target_temperature_c = p.tech.optimum_stack_temperature_c(current_a)
    target_temperature_k = 273.15 + target_temperature_c
    floor_flow_slpm = p.stack.coolant_air_target_slpm(current_a)

    rows: list[dict[str, Any]] = []
    for inlet_temperature_c in args.inlet_temperatures_c:
        inlet_temperature_k = 273.15 + inlet_temperature_c
        target = open_cathode_airflow_target(
            heat_rejection_w=heat_rejection_w,
            inlet_temperature_k=inlet_temperature_k,
            target_stack_temperature_k=target_temperature_k,
            stoichiometric_floor_slpm=floor_flow_slpm,
        )
        if target.target_air_flow_slpm is None:
            continue

        flow = target.target_air_flow_slpm
        tau_s = lumped_thermal_time_constant_s(
            thermal_mass_j_k=stack.thermal_mass_j_k,
            air_flow_slpm=flow,
        )
        for offset_k in args.initial_offsets_k:
            initial_temperature_k = inlet_temperature_k + offset_k
            analytic = lumped_temperature_analytic_k(
                time_s=args.duration_s,
                initial_stack_temperature_k=initial_temperature_k,
                heat_rejection_w=heat_rejection_w,
                inlet_temperature_k=inlet_temperature_k,
                air_flow_slpm=flow,
                thermal_mass_j_k=stack.thermal_mass_j_k,
            )
            for dt_s in args.time_steps_s:
                numerical = integrate_constant_case(
                    initial_temperature_k=initial_temperature_k,
                    inlet_temperature_k=inlet_temperature_k,
                    heat_rejection_w=heat_rejection_w,
                    air_flow_slpm=flow,
                    thermal_mass_j_k=stack.thermal_mass_j_k,
                    duration_s=args.duration_s,
                    dt_s=dt_s,
                )
                error_k = numerical - analytic
                rows.append(
                    {
                        "inlet_temperature_c": inlet_temperature_c,
                        "initial_offset_k": offset_k,
                        "dt_s": dt_s,
                        "air_flow_slpm": flow,
                        "thermal_time_constant_s": tau_s,
                        "analytic_final_temperature_c": analytic - 273.15,
                        "numerical_final_temperature_c": numerical - 273.15,
                        "error_k": error_k,
                        "absolute_error_k": abs(error_k),
                    }
                )

    max_error = max(float(row["absolute_error_k"]) for row in rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "v08-lumped-thermal-validation",
                "duration_s": args.duration_s,
                "thermal_mass_j_k": stack.thermal_mass_j_k,
                "max_absolute_error_k": max_error,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"Tin={row['inlet_temperature_c']:.1f} C "
            f"dT0={row['initial_offset_k']:.1f} K "
            f"dt={row['dt_s']:.2f} s "
            f"tau={row['thermal_time_constant_s']:.1f} s "
            f"error={row['error_k']:+.4f} K",
            flush=True,
        )
    print(f"max absolute error={max_error:.5f} K")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
