"""Scan Newton damping for the first V1.0 stationary reaction level."""

from __future__ import annotations

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


def _run_case(
    args: tuple[CathodeParameters, float, float],
) -> dict[str, Any]:
    params, oxygen_feed, damping = args
    result = solve_stationary(
        params=params,
        nx=4,
        ny=4,
        nz=16,
        oxygen_feed_concentration=oxygen_feed,
        newton_tolerance=1e-6,
        max_newton_iterations=40,
        newton_damping=damping,
        reaction_scales=(1e-6,),
    )
    stage = result["continuation_history"][-1]
    norm_history = stage["perturbation_norm_history"]
    return {
        "damping": damping,
        "converged": result["converged"],
        "iterations": result["newton_iterations"],
        "final_norm": result["perturbation_norm"],
        "minimum_norm": min(norm_history),
        "minimum_norm_iteration": norm_history.index(min(norm_history)) + 1,
        "final_min_c_o2_mol_m3": result["min_c_o2_mol_m3"],
        "norm_history": norm_history,
        "min_c_o2_history": stage["min_c_o2_history"],
    }


def main() -> None:
    base = CathodeParameters()
    current_a = 26.04
    inlet_temperature_c = 20.0
    ntu = 3.0
    xi = 0.5
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
    oxygen_feed = float(np.interp(xi, profile.streamwise_fraction, concentration))

    dampings = [0.25, 0.5, 0.75, 1.0]
    tasks = [(params, oxygen_feed, damping) for damping in dampings]
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        rows = list(executor.map(_run_case, tasks))

    rows.sort(key=lambda row: float(row["damping"]))
    output = {
        "schema_version": 1,
        "model": "v10-stationary-damping-scan",
        "reaction_scale": 1e-6,
        "grid": [4, 4, 16],
        "max_newton_iterations": 40,
        "rows": rows,
    }

    output_path = Path("results/quick-v10-stationary-damping-scan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    for row in rows:
        print(
            f"damping={row['damping']:.2f} "
            f"converged={row['converged']} "
            f"min_norm={row['minimum_norm']:.3e} "
            f"at={row['minimum_norm_iteration']} "
            f"final_norm={row['final_norm']:.3e} "
            f"cO2_min={row['final_min_c_o2_mol_m3']:.6g}",
            flush=True,
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
