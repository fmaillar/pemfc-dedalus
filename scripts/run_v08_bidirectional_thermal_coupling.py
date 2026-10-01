"""Iterate the V0.8 thermal/N2 coupling to a self-consistent trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.anode_nitrogen import humid_air_nitrogen_partial_pressure_pa
from pemfc_dedalus.ballard_1020acs import UserStackConfiguration
from pemfc_dedalus.gas_permeability import (
    arrhenius_permeability,
    barrer_to_si_permeability,
)
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from pemfc_dedalus.thermal import (
    advance_lumped_stack_temperature_k,
    open_cathode_airflow_target,
)
from scripts.run_v07_anode_h2 import interpolate_flux_and_current, load_h2_closure
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime
from scripts.run_v07_n2_catalano_calibrated import (
    DRY_N2_ACTIVATION_ENERGY_J_MOL,
    DRY_REFERENCE_PERMEABILITY_BARRER,
    REFERENCE_TEMPERATURE_K,
)
from scripts.run_v07_n2_catalano_membrane_transport import (
    DEFAULT_ACTIVATION_ENERGY_J_MOL,
    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
)
from scripts.run_v07_n2_catalano_motupally import (
    build_motupally_lookup_table,
    make_tabulated_permeance_model,
)


def read_temperature_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("temperature trajectory must contain at least two rows")
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    temperatures = np.asarray(
        [float(row["stack_temperature_c"]) + 273.15 for row in rows],
        dtype=float,
    )
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("temperature trajectory time must increase")
    return times, temperatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure-csv",
        type=Path,
        default=Path("results/v06-rha-sensitivity.csv"),
    )
    parser.add_argument(
        "--initial-temperature-csv",
        type=Path,
        default=Path("results/quick-v08-lumped-thermal-dynamics.csv"),
    )
    parser.add_argument("--regime", default="nominal")
    parser.add_argument("--feedback-exponent", type=float, default=1.0)
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--stop-time", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--lookup-rh-points", type=int, default=21)
    parser.add_argument("--lookup-current-points", type=int, default=21)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--temperature-tolerance-k", type=float, default=0.01)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v08-bidirectional-thermal-coupling.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v08-bidirectional-thermal-coupling.csv"),
    )
    args = parser.parse_args()

    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")
    if args.temperature_tolerance_k <= 0.0:
        parser.error("--temperature-tolerance-k must be positive")

    p = CathodeParameters()
    stack = UserStackConfiguration()
    closure = load_h2_closure(args.closure_csv)
    if args.regime not in closure:
        parser.error(f"unknown regime: {args.regime}")

    time_axis, temperature_k = read_temperature_trajectory(
        args.initial_temperature_csv
    )
    if abs(float(time_axis[-1]) - args.stop_time) > 1.0e-9:
        parser.error("temperature trajectory duration must match --stop-time")

    patch_area = p.length_x * p.length_y
    _, patch_reference_current = interpolate_flux_and_current(
        0.0,
        closure["nominal"],
    )
    scaling = infer_active_area_scaling(
        patch_area_m2=patch_area,
        patch_reference_current_a=patch_reference_current,
        cell_reference_current_a=p.stack_current_a,
    )
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )

    current_density_values = [
        float(value) * scaling.area_scale_factor / scaling.inferred_active_area_m2
        for value in closure[args.regime][2]
    ]
    current_min = min(current_density_values)
    current_max = max(current_density_values)
    current_margin = max(0.1 * (current_max - current_min), 50.0)
    rh_axis = np.linspace(0.0, 1.0, args.lookup_rh_points)
    current_axis = np.linspace(
        max(0.0, current_min - current_margin),
        current_max + current_margin,
        args.lookup_current_points,
    )

    baseline_lookup = build_motupally_lookup_table(
        relative_humidity_axis=rh_axis,
        current_density_axis=current_axis,
        jobs=args.jobs,
        activation_energy_j_mol=DEFAULT_ACTIVATION_ENERGY_J_MOL,
        water_partial_molar_volume_cm3_mol=(
            DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL
        ),
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
        stack_temperature_k=p.stack_temperature,
    )
    baseline_permeance_model = make_tabulated_permeance_model(
        rh_axis,
        current_axis,
        baseline_lookup,
    )

    dry_reference_si = barrer_to_si_permeability(
        DRY_REFERENCE_PERMEABILITY_BARRER
    )
    baseline_dry_permeability = arrhenius_permeability(
        dry_reference_si,
        p.stack_temperature,
        REFERENCE_TEMPERATURE_K,
        DRY_N2_ACTIVATION_ENERGY_J_MOL,
        p.gas_constant,
    )

    def thermal_permeance_model(
        relative_humidity: float,
        current_density_a_m2: float,
        local_temperature_k: float,
    ) -> float:
        dry_at_temperature = arrhenius_permeability(
            dry_reference_si,
            local_temperature_k,
            REFERENCE_TEMPERATURE_K,
            DRY_N2_ACTIVATION_ENERGY_J_MOL,
            p.gas_constant,
        )
        return (
            baseline_permeance_model(
                relative_humidity,
                current_density_a_m2,
            )
            * dry_at_temperature
            / baseline_dry_permeability
        )

    def cathode_n2_pressure_model(local_temperature_k: float) -> float:
        return humid_air_nitrogen_partial_pressure_pa(
            total_pressure_pa=p.pressure,
            oxygen_dry_mole_fraction=p.oxygen_mole_fraction,
            relative_humidity=p.relative_humidity,
            saturation_water_pressure_pa=water_saturation_pressure_pa(
                local_temperature_k
            ),
        )

    common: dict[str, Any] = {
        "initial_rh": 0.0,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "write_every": args.write_every,
        "volume_m3": p.anode_gas_volume_m3,
        "temperature_k": p.stack_temperature,
        "gas_constant_j_mol_k": p.gas_constant,
        "faraday_c_mol": p.faraday,
        "active_area_m2": scaling.inferred_active_area_m2,
        "fixed_charge_mol_m3": fixed_charge,
        "target_total_pressure_pa": p.anode_target_total_pressure_pa,
        "ambient_pressure_pa": p.pressure,
        "purge_interval_as": p.tech.purge_interval_as,
        "purge_duration_s": p.tech.purge_duration_max_s,
        "purge_reference_flow_slpm": p.tech.purge_rate_min_slpm_per_cell,
        "current_scale_factor": scaling.area_scale_factor,
        "n2_crossover_flux_mol_m2_s": 0.0,
        "hydrogen_feedback_exponent": args.feedback_exponent,
        "n2_state_temperature_permeance_model": thermal_permeance_model,
        "cathode_n2_partial_pressure_pa": cathode_n2_pressure_model(
            p.stack_temperature
        ),
        "cathode_n2_partial_pressure_model": cathode_n2_pressure_model,
    }

    inlet_temperature_k = 273.15 + args.inlet_temperature_c
    initial_temperature_k = float(temperature_k[0])
    initial_trajectory_k = temperature_k.copy()
    iteration_summaries: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    converged = False

    for iteration in range(1, args.max_iterations + 1):
        previous_temperature_k = temperature_k.copy()

        def temperature_model(
            time_s: float,
            trajectory_k: np.ndarray = previous_temperature_k,
        ) -> float:
            return float(
                np.interp(
                    time_s,
                    time_axis,
                    trajectory_k,
                    left=trajectory_k[0],
                    right=trajectory_k[-1],
                )
            )

        rows, summary = simulate_nitrogen_regime(
            args.regime,
            closure[args.regime],
            temperature_model_k=temperature_model,
            **common,
        )
        row_times = np.asarray(
            [float(row["time_s"]) for row in rows],
            dtype=float,
        )
        if row_times.shape != time_axis.shape or not np.allclose(
            row_times,
            time_axis,
        ):
            raise ValueError(
                "dynamic output grid does not match thermal trajectory grid"
            )

        new_temperature_k = np.empty_like(previous_temperature_k)
        new_temperature_k[0] = initial_temperature_k

        for index in range(len(rows) - 1):
            current_a = float(rows[index]["cell_current_a"])
            vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
            heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
            target_temperature_k = (
                273.15 + p.tech.optimum_stack_temperature_c(current_a)
            )
            floor_flow_slpm = p.stack.coolant_air_target_slpm(current_a)
            airflow = open_cathode_airflow_target(
                heat_rejection_w=heat_rejection_w,
                inlet_temperature_k=inlet_temperature_k,
                target_stack_temperature_k=target_temperature_k,
                stoichiometric_floor_slpm=floor_flow_slpm,
            )
            if airflow.target_air_flow_slpm is None:
                raise ValueError(
                    "thermal target became unreachable during iteration"
                )

            dt_s = float(time_axis[index + 1] - time_axis[index])
            new_temperature_k[index + 1] = (
                advance_lumped_stack_temperature_k(
                    stack_temperature_k=new_temperature_k[index],
                    inlet_temperature_k=inlet_temperature_k,
                    heat_rejection_w=heat_rejection_w,
                    air_flow_slpm=airflow.target_air_flow_slpm,
                    thermal_mass_j_k=stack.thermal_mass_j_k,
                    dt_s=dt_s,
                )
            )

        delta_k = np.abs(new_temperature_k - previous_temperature_k)
        max_delta_k = float(np.max(delta_k))
        rms_delta_k = float(np.sqrt(np.mean(delta_k**2)))
        iteration_summaries.append(
            {
                "iteration": iteration,
                "max_temperature_change_k": max_delta_k,
                "rms_temperature_change_k": rms_delta_k,
                "final_temperature_c": float(
                    new_temperature_k[-1] - 273.15
                ),
                "mean_cell_current_a": float(summary["mean_cell_current_a"]),
                "mean_n2_crossover_flux_mol_m2_s": float(
                    summary["mean_n2_crossover_flux_mol_m2_s"]
                ),
                "mean_purge_period_s": float(
                    summary["mean_purge_period_s"]
                ),
            }
        )

        temperature_k = new_temperature_k
        final_rows = rows
        print(
            f"iteration={iteration} "
            f"max_dT={max_delta_k:.6f} K "
            f"rms_dT={rms_delta_k:.6f} K "
            f"Tfinal={temperature_k[-1] - 273.15:.4f} C "
            f"Imean={summary['mean_cell_current_a']:.5f} A",
            flush=True,
        )

        if max_delta_k <= args.temperature_tolerance_k:
            converged = True
            break

    total_delta_k = temperature_k - initial_trajectory_k
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(final_rows):
        output_rows.append(
            {
                "time_s": float(row["time_s"]),
                "cell_current_a": float(row["cell_current_a"]),
                "initial_temperature_c": float(
                    initial_trajectory_k[index] - 273.15
                ),
                "converged_temperature_c": float(
                    temperature_k[index] - 273.15
                ),
                "temperature_change_k": float(total_delta_k[index]),
            }
        )

    output = {
        "schema_version": 1,
        "model": "v08-bidirectional-thermal-coupling",
        "converged": converged,
        "temperature_tolerance_k": args.temperature_tolerance_k,
        "iterations_completed": len(iteration_summaries),
        "max_iterations": args.max_iterations,
        "initial_final_temperature_c": float(
            initial_trajectory_k[-1] - 273.15
        ),
        "converged_final_temperature_c": float(
            temperature_k[-1] - 273.15
        ),
        "max_total_temperature_change_k": float(
            np.max(np.abs(total_delta_k))
        ),
        "rms_total_temperature_change_k": float(
            np.sqrt(np.mean(total_delta_k**2))
        ),
        "iterations": iteration_summaries,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"converged={converged} "
        f"iterations={len(iteration_summaries)} "
        f"max_total_dT={output['max_total_temperature_change_k']:.6f} K",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
