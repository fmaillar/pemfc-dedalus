"""Diagnose which variables dominate the stationary Newton norm."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pemfc_dedalus.cathode_airflow import ideal_gas_species_concentration_mol_m3
from pemfc_dedalus.cathode_airflow_1d import solve_streamwise_finite_thermal_profile
from pemfc_dedalus.cathode_electrochem_stationary_3d import solve_stationary
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


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

    result = solve_stationary(
        params=params,
        nx=4,
        ny=4,
        nz=16,
        oxygen_feed_concentration=oxygen_feed,
        newton_tolerance=1e-12,
        max_newton_iterations=40,
        newton_damping=0.25,
        reaction_scales=(1e-6,),
    )
    stage = result["continuation_history"][-1]
    histories = stage["component_norm_history"]

    final_components = histories[-1]
    minimum_components = {
        name: min(step[name] for step in histories)
        for name in final_components
    }
    dominant_final = max(final_components, key=final_components.get)

    output = {
        "schema_version": 1,
        "model": "v10-stationary-component-norms",
        "reaction_scale": 1e-6,
        "damping": 0.25,
        "grid": [4, 4, 16],
        "iterations": len(histories),
        "total_norm_history": stage["perturbation_norm_history"],
        "component_norm_history": histories,
        "final_component_norms": final_components,
        "minimum_component_norms": minimum_components,
        "dominant_final_component": dominant_final,
        "final_min_c_o2_mol_m3": result["min_c_o2_mol_m3"],
    }

    output_path = Path("results/quick-v10-stationary-component-norms.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    print(f"dominant_final_component={dominant_final}", flush=True)
    for name, value in sorted(
        final_components.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{name}: final={value:.3e} minimum={minimum_components[name]:.3e}",
            flush=True,
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
