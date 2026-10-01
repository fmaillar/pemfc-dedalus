"""Project thermo-coupled O2 profiles onto electrochemical slices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.cathode_airflow import ideal_gas_species_concentration_mol_m3
from pemfc_dedalus.cathode_airflow_1d import (
    solve_streamwise_finite_thermal_profile,
)
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntu-values", nargs="+", type=float, default=[1.0, 3.0, 5.0])
    parser.add_argument("--slice-xi", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--currents-a", nargs="+", type=float, default=[7.3, 14.5, 26.04])
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
        default=Path("results/quick-v09-electrochem-slices.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v09-electrochem-slices.csv"),
    )
    args = parser.parse_args()

    if any(not 0.0 <= xi <= 1.0 for xi in args.slice_xi):
        raise ValueError("slice-xi values must lie in [0, 1]")

    p = CathodeParameters()
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
                concentration = ideal_gas_species_concentration_mol_m3(
                    mole_fraction=solved.oxygen_mole_fraction,
                    pressure_pa=p.pressure,
                    temperature_k=solved.air_temperature_k,
                    gas_constant_j_mol_k=p.gas_constant,
                )
                reference_concentration = (
                    p.oxygen_mole_fraction
                    * p.pressure
                    / (p.gas_constant * target_temperature_k)
                )

                for xi in args.slice_xi:
                    local_c = float(
                        np.interp(
                            xi,
                            solved.streamwise_fraction,
                            concentration,
                        )
                    )
                    oxygen_activity = local_c / reference_concentration
                    kinetic_factor = oxygen_activity ** p.oxygen_reaction_order
                    rows.append(
                        {
                            "current_a": current_a,
                            "inlet_temperature_c": inlet_temperature_c,
                            "ntu": ntu,
                            "slice_xi": xi,
                            "local_o2_concentration_mol_m3": local_c,
                            "reference_o2_concentration_mol_m3": (
                                reference_concentration
                            ),
                            "oxygen_activity": oxygen_activity,
                            "kinetic_factor": kinetic_factor,
                        }
                    )

    by_case: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["current_a"]),
            float(row["inlet_temperature_c"]),
            float(row["ntu"]),
        )
        by_case.setdefault(key, []).append(row)

    max_span = 0.0
    for case_rows in by_case.values():
        factors = [float(row["kinetic_factor"]) for row in case_rows]
        max_span = max(max_span, max(factors) - min(factors))

    output = {
        "schema_version": 1,
        "model": "v09-electrochemical-slice-forcing",
        "slice_xi": args.slice_xi,
        "max_kinetic_factor_span": max_span,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"rows={len(rows)} "
        f"max_kinetic_factor_span={100.0 * max_span:.2f} percentage-points",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
