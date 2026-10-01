"""Run Picard-initialized globalized inverse Butler-Volmer stationary solve."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pemfc_dedalus.cathode_airflow import ideal_gas_species_concentration_mol_m3
from pemfc_dedalus.cathode_airflow_1d import solve_streamwise_finite_thermal_profile
from pemfc_dedalus.cathode_electrochem_inverse_bv_3d import (
    solve_stationary_inverse_bv,
)
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


def main() -> None:
    base = CathodeParameters()
    current_a = 26.04
    inlet_temperature_c = 20.0
    ntu = 3.0
    xi = 0.5
    nx, ny, nz = 4, 4, 16

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

    result = solve_stationary_inverse_bv(
        params=params,
        nx=nx,
        ny=ny,
        nz=nz,
        oxygen_feed_concentration=oxygen_feed,
        max_newton_iterations=120,
        globalized=True,
        residual_tolerance=1e-8,
        max_backtracks=12,
        picard_iterations=160,
        picard_relaxation=0.25,
        picard_tolerance=1e-8,
    )

    area_m2 = params.length_x * params.length_y
    current_density = result["total_reaction_current"] / area_m2
    target_current_density = base.membrane_current_density
    relative_current_error = (
        current_density - target_current_density
    ) / target_current_density

    output = {
        "schema_version": 1,
        "model": "v10-stationary-inverse-butler-volmer-picard-globalized",
        "current_a": current_a,
        "voltage_v": voltage_v,
        "slice_xi": xi,
        "grid": [nx, ny, nz],
        "oxygen_feed_concentration_mol_m3": oxygen_feed,
        "target_current_density_a_m2": target_current_density,
        "current_density_a_m2": current_density,
        "relative_current_error": relative_current_error,
        **result,
    }

    output_path = Path(
        "results/quick-v10-stationary-inverse-bv-picard.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    print(
        f"picard={result['picard_iterations']} "
        f"relaxation={result['picard_relaxation']:.3g}",
        flush=True,
    )
    for picard_step in result["picard_history"]:
        print(
            f"picard_iter={picard_step['iteration']} "
            f"fp_res={picard_step['fixed_point_residual']:.3e} "
            f"dq_rel={picard_step['relative_q_change']:.3e} "
            f"alpha={picard_step['damping']:.6g} "
            f"backtracks={picard_step['backtracks']} "
            f"cO2_min={picard_step['min_c_o2_mol_m3']:.6g} "
            f"qhat=[{picard_step['min_q_hat']:.6g},"
            f"{picard_step['max_q_hat']:.6g}]",
            flush=True,
        )
    print(
        f"converged={result['converged']} "
        f"picard_converged={result['picard_converged']} "
        f"fp_res={result['picard_fixed_point_residual']:.3e} "
        f"newton={result['newton_iterations']} "
        f"residual={result['residual_merit']:.3e}",
        flush=True,
    )
    print(
        f"cO2_min={result['min_c_o2_mol_m3']:.6g} mol/m3 "
        f"qhat_min={result['min_q_hat']:.6g} "
        f"j={current_density:.6g} A/m2 "
        f"error={100.0 * relative_current_error:+.3f}%",
        flush=True,
    )
    print(
        f"pre_newton_residual={result['pre_newton_residual_merit']:.6e}",
        flush=True,
    )
    print("pre-Newton normalized residual components:", flush=True)
    for name, value in result["pre_newton_residual_components"].items():
        print(f"  {name}={value:.6e}", flush=True)
    print("final normalized residual components:", flush=True)
    for name, value in result["residual_components"].items():
        print(f"  {name}={value:.6e}", flush=True)
    for newton_step in result["globalization_history"]:
        print(
            f"iter={newton_step['iteration']} "
            f"residual={newton_step['residual_start']:.3e}"
            f"->{newton_step['residual_final']:.3e} "
            f"alpha={newton_step['damping']:.6g} "
            f"backtracks={newton_step['backtracks']} "
            f"cO2_min={newton_step['min_c_o2_mol_m3']:.6g} "
            f"qhat_min={newton_step['min_q_hat']:.6g}",
            flush=True,
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
