"""Screen thermo-coupled local cathode O2 gas state for V0.9."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.cathode_airflow import ideal_gas_species_concentration_mol_m3
from pemfc_dedalus.cathode_airflow_1d import (\n    solve_streamwise_finite_thermal_profile,\n)
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntu-values", nargs="+", type=float, default=[1.0, 3.0, 5.0])
    parser.add_argument(\n        "--currents-a", nargs="+", type=float, default=[7.3, 14.5, 26.04]\n    )
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
        default=Path("results/quick-v09-local-o2-state.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v09-local-o2-state.csv"),
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

                local_concentration = ideal_gas_species_concentration_mol_m3(
                    mole_fraction=solved.oxygen_mole_fraction,
                    pressure_pa=p.pressure,
                    temperature_k=solved.air_temperature_k,
                    gas_constant_j_mol_k=p.gas_constant,
                )
                isothermal_concentration = ideal_gas_species_concentration_mol_m3(
                    mole_fraction=solved.oxygen_mole_fraction,
                    pressure_pa=p.pressure,
                    temperature_k=target_temperature_k,
                    gas_constant_j_mol_k=p.gas_constant,
                )

                inlet_c = float(local_concentration[0])
                outlet_c = float(local_concentration[-1])
                relative_drop = (inlet_c - outlet_c) / inlet_c

                iso_inlet_c = float(isothermal_concentration[0])
                iso_outlet_c = float(isothermal_concentration[-1])
                composition_only_drop = (\n                    iso_inlet_c - iso_outlet_c\n                ) / iso_inlet_c

                summaries.append(
                    {
                        "current_a": current_a,
                        "inlet_temperature_c": inlet_temperature_c,
                        "ntu": ntu,
                        "status": airflow.status,
                        "active_constraint": airflow.active_constraint,
                        "inlet_o2_concentration_mol_m3": inlet_c,
                        "outlet_o2_concentration_mol_m3": outlet_c,
                        "relative_o2_concentration_drop": relative_drop,
                        "isothermal_composition_only_drop": composition_only_drop,
                        "additional_thermal_drop": relative_drop - composition_only_drop,
                    }
                )

                for index, xi in enumerate(solved.streamwise_fraction):
                    rows.append(
                        {
                            "current_a": current_a,
                            "inlet_temperature_c": inlet_temperature_c,
                            "ntu": ntu,
                            "streamwise_fraction": float(xi),
                            "oxygen_mole_fraction": float(
                                solved.oxygen_mole_fraction[index]
                            ),
                            "air_temperature_c": float(
                                solved.air_temperature_k[index] - 273.15
                            ),
                            "oxygen_concentration_mol_m3": float(
                                local_concentration[index]
                            ),
                            "isothermal_oxygen_concentration_mol_m3": float(
                                isothermal_concentration[index]
                            ),
                        }
                    )

    reachable = [\n        row for row in summaries if "relative_o2_concentration_drop" in row\n    ]
    max_drop = max(\n        float(row["relative_o2_concentration_drop"]) for row in reachable\n    )
    max_thermal = max(float(row["additional_thermal_drop"]) for row in reachable)

    output = {
        "schema_version": 1,
        "model": "v09-local-thermo-coupled-o2-state",
        "ntu_values": args.ntu_values,
        "points": args.points,
        "max_relative_o2_concentration_drop": max_drop,
        "max_additional_thermal_drop": max_thermal,
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
        f"max_cO2_drop={100.0 * max_drop:.2f}% "
        f"max_extra_thermal_drop={100.0 * max_thermal:.2f}%",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
