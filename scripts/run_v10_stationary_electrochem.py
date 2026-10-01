"""Run nominal V1.0 stationary electrochemistry on three streamwise slices."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.cathode_airflow import ideal_gas_species_concentration_mol_m3
from pemfc_dedalus.cathode_airflow_1d import solve_streamwise_finite_thermal_profile
from pemfc_dedalus.cathode_electrochem_stationary_3d import solve_stationary
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


def simpson_mean(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("Simpson mean requires exactly three values")
    return (values[0] + 4.0 * values[1] + values[2]) / 6.0


def main() -> None:
    base = CathodeParameters()
    current_a = 26.04
    inlet_temperature_c = 20.0
    ntu = 3.0
    voltage_v = base.tech.bol_typical_cell_voltage_v(current_a)
    target_temperature_k = 273.15 + base.tech.optimum_stack_temperature_c(current_a)
    inlet_temperature_k = 273.15 + inlet_temperature_c

    params = replace(
        base,
        stack_current_a=current_a,
        stack_temperature=target_temperature_k,
        cathode_solid_potential=voltage_v,
    )

    heat_rejection_w = params.stack.heat_rejection_w(current_a, voltage_v)
    airflow = open_cathode_airflow_target(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        target_stack_temperature_k=target_temperature_k,
        stoichiometric_floor_slpm=params.stack.coolant_air_target_slpm(current_a),
    )
    if airflow.target_air_flow_slpm is None:
        raise RuntimeError("nominal point is thermally unreachable")

    profile = solve_streamwise_finite_thermal_profile(
        current_a=current_a,
        total_air_flow_slpm=airflow.target_air_flow_slpm,
        n_cells=params.stack.n_cells,
        oxygen_mole_fraction=params.oxygen_mole_fraction,
        faraday_c_mol=params.faraday,
        inlet_temperature_k=inlet_temperature_k,
        stack_temperature_k=target_temperature_k,
        ntu=ntu,
        points=64,
    )
    concentration = ideal_gas_species_concentration_mol_m3(
        mole_fraction=profile.oxygen_mole_fraction,
        pressure_pa=params.pressure,
        temperature_k=profile.air_temperature_k,
        gas_constant_j_mol_k=params.gas_constant,
    )

    xis = [0.0, 0.5, 1.0]
    local_feed = {
        xi: float(np.interp(xi, profile.streamwise_fraction, concentration))
        for xi in xis
    }

    def run_slice(xi: float) -> dict[str, Any]:
        result = solve_stationary(
            params=params,
            nx=8,
            ny=8,
            nz=32,
            oxygen_feed_concentration=local_feed[xi],
            newton_tolerance=1e-8,
            max_newton_iterations=30,
        )
        return {
            "slice_xi": xi,
            "oxygen_feed_concentration_mol_m3": local_feed[xi],
            **result,
        }

    with ProcessPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(run_slice, xis))

    rows.sort(key=lambda row: float(row["slice_xi"]))
    area_m2 = params.length_x * params.length_y
    local_current_density = [
        float(row["total_reaction_current"]) / area_m2 for row in rows
    ]
    mean_current_density = simpson_mean(local_current_density)
    target_current_density = base.membrane_current_density

    output = {
        "schema_version": 1,
        "model": "v10-stationary-electrochemistry",
        "current_a": current_a,
        "voltage_v": voltage_v,
        "target_current_density_a_m2": target_current_density,
        "mean_current_density_a_m2": mean_current_density,
        "relative_current_error": (
            mean_current_density - target_current_density
        ) / target_current_density,
        "all_converged": all(bool(row["converged"]) for row in rows),
        "all_positive_o2": all(float(row["min_c_o2_mol_m3"]) > 0.0 for row in rows),
        "all_positive_orr": all(float(row["min_j_orr_a_m3"]) >= 0.0 for row in rows),
        "rows": rows,
    }

    output_json = Path("results/quick-v10-stationary-electrochem.json")
    output_csv = Path("results/quick-v10-stationary-electrochem.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n")

    csv_rows = []
    for row, current_density in zip(rows, local_current_density, strict=True):
        csv_rows.append(
            {
                **row,
                "current_density_a_m2": current_density,
            }
        )
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"all_converged={output['all_converged']} "
        f"all_positive_o2={output['all_positive_o2']} "
        f"all_positive_orr={output['all_positive_orr']}",
        flush=True,
    )
    relative_current_error = (
        mean_current_density - target_current_density
    ) / target_current_density
    print(
        f"j_bar={mean_current_density:.2f} A/m2 "
        f"error={100.0 * relative_current_error:+.3f}%",
        flush=True,
    )
    for row in rows:
        print(
            f"xi={row['slice_xi']:.1f} "
            f"newton={row['newton_iterations']} "
            f"norm={row['perturbation_norm']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
