"""Couple V0.7 anode dynamics to the V0.8 instantaneous airflow law."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.anode_nitrogen import humid_air_nitrogen_partial_pressure_pa
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from pemfc_dedalus.thermal import open_cathode_airflow_target
from scripts.run_v07_anode_h2 import (
    interpolate_flux_and_current,
    load_h2_closure,
)
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime
from scripts.run_v07_n2_catalano_membrane_transport import (
    DEFAULT_ACTIVATION_ENERGY_J_MOL,
    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
)
from scripts.run_v07_n2_catalano_motupally import (
    build_motupally_lookup_table,
    make_tabulated_permeance_model,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty thermal-dynamic result")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure-csv",
        type=Path,
        default=Path("results/v06-rha-sensitivity.csv"),
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v08-thermal-dynamic-coupling.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v08-thermal-dynamic-coupling.csv"),
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.lookup_rh_points < 3 or args.lookup_current_points < 3:
        parser.error("lookup axes must each contain at least three points")

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
    if args.regime not in closure:
        parser.error(f"unknown regime: {args.regime}")

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
    saturation_pressure = water_saturation_pressure_pa(p.stack_temperature)
    cathode_n2_partial_pressure = humid_air_nitrogen_partial_pressure_pa(
        total_pressure_pa=p.pressure,
        oxygen_dry_mole_fraction=p.oxygen_mole_fraction,
        relative_humidity=p.relative_humidity,
        saturation_water_pressure_pa=saturation_pressure,
    )

    current_density_values = [
        float(patch_current_a)
        * scaling.area_scale_factor
        / scaling.inferred_active_area_m2
        for patch_current_a in closure[args.regime][2]
    ]
    current_min = min(current_density_values)
    current_max = max(current_density_values)
    margin = max(0.1 * (current_max - current_min), 50.0)
    rh_axis = np.linspace(0.0, 1.0, args.lookup_rh_points)
    current_axis = np.linspace(
        max(0.0, current_min - margin),
        current_max + margin,
        args.lookup_current_points,
    )

    print(
        "Precomputing V0.7 Motupally lookup for thermal coupling: "
        f"{args.lookup_rh_points}x{args.lookup_current_points} "
        f"with {min(args.jobs, args.lookup_rh_points)} workers",
        flush=True,
    )
    lookup = build_motupally_lookup_table(
        relative_humidity_axis=rh_axis,
        current_density_axis=current_axis,
        jobs=args.jobs,
        activation_energy_j_mol=DEFAULT_ACTIVATION_ENERGY_J_MOL,
        water_partial_molar_volume_cm3_mol=(
            DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL
        ),
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )
    permeance_model = make_tabulated_permeance_model(
        rh_axis,
        current_axis,
        lookup,
    )

    dynamic_rows, dynamic_summary = simulate_nitrogen_regime(
        args.regime,
        closure[args.regime],
        initial_rh=0.0,
        stop_time_s=args.stop_time,
        dt_s=args.dt,
        write_every=args.write_every,
        volume_m3=p.anode_gas_volume_m3,
        temperature_k=p.stack_temperature,
        gas_constant_j_mol_k=p.gas_constant,
        faraday_c_mol=p.faraday,
        active_area_m2=scaling.inferred_active_area_m2,
        fixed_charge_mol_m3=fixed_charge,
        target_total_pressure_pa=p.anode_target_total_pressure_pa,
        ambient_pressure_pa=p.pressure,
        purge_interval_as=p.tech.purge_interval_as,
        purge_duration_s=p.tech.purge_duration_max_s,
        purge_reference_flow_slpm=p.tech.purge_rate_min_slpm_per_cell,
        current_scale_factor=scaling.area_scale_factor,
        n2_crossover_flux_mol_m2_s=0.0,
        hydrogen_feedback_exponent=args.feedback_exponent,
        n2_state_permeance_model=permeance_model,
        cathode_n2_partial_pressure_pa=cathode_n2_partial_pressure,
    )

    inlet_temperature_k = 273.15 + args.inlet_temperature_c
    thermal_rows: list[dict[str, Any]] = []

    for row in dynamic_rows:
        current_a = float(row["cell_current_a"])
        vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
        heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
        target_temperature_c = p.tech.optimum_stack_temperature_c(current_a)
        floor_flow_slpm = p.stack.coolant_air_target_slpm(current_a)
        target = open_cathode_airflow_target(
            heat_rejection_w=heat_rejection_w,
            inlet_temperature_k=inlet_temperature_k,
            target_stack_temperature_k=273.15 + target_temperature_c,
            stoichiometric_floor_slpm=floor_flow_slpm,
        )
        thermal_rows.append(
            {
                "time_s": float(row["time_s"]),
                "purge_open": bool(row["purge_open"]),
                "purge_count": int(row["purge_count"]),
                "cell_current_a": current_a,
                "vcell_bol_typ_v": vcell_v,
                "heat_rejection_w": heat_rejection_w,
                "target_stack_temperature_c": target_temperature_c,
                "airflow_status": target.status,
                "active_constraint": target.active_constraint,
                "stoichiometric_floor_slpm": floor_flow_slpm,
                "thermal_required_slpm": target.thermal_required_slpm,
                "target_air_flow_slpm": target.target_air_flow_slpm,
            }
        )

    reachable = [
        row for row in thermal_rows if row["target_air_flow_slpm"] is not None
    ]
    target_flows = [
        float(row["target_air_flow_slpm"]) for row in reachable
    ]
    currents = [float(row["cell_current_a"]) for row in thermal_rows]
    heats = [float(row["heat_rejection_w"]) for row in thermal_rows]
    purge_flows = [
        float(row["target_air_flow_slpm"])
        for row in reachable
        if row["purge_open"]
    ]
    nonpurge_flows = [
        float(row["target_air_flow_slpm"])
        for row in reachable
        if not row["purge_open"]
    ]

    summary = {
        "schema_version": 1,
        "model": "v08-thermal-dynamic-coupling",
        "regime": args.regime,
        "feedback_exponent": args.feedback_exponent,
        "inlet_temperature_c": args.inlet_temperature_c,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "write_every": args.write_every,
        "lookup_rh_points": args.lookup_rh_points,
        "lookup_current_points": args.lookup_current_points,
        "dynamic_summary": dynamic_summary,
        "min_cell_current_a": min(currents),
        "max_cell_current_a": max(currents),
        "current_range_a": max(currents) - min(currents),
        "min_heat_rejection_w": min(heats),
        "max_heat_rejection_w": max(heats),
        "min_target_air_flow_slpm": min(target_flows),
        "max_target_air_flow_slpm": max(target_flows),
        "target_air_flow_range_slpm": max(target_flows) - min(target_flows),
        "mean_target_air_flow_slpm": float(np.mean(target_flows)),
        "mean_purge_target_air_flow_slpm": (
            float(np.mean(purge_flows)) if purge_flows else None
        ),
        "mean_nonpurge_target_air_flow_slpm": (
            float(np.mean(nonpurge_flows)) if nonpurge_flows else None
        ),
        "unreachable_samples": sum(
            row["target_air_flow_slpm"] is None for row in thermal_rows
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_csv, thermal_rows)

    print(
        f"I={summary['min_cell_current_a']:.3f}.."
        f"{summary['max_cell_current_a']:.3f} A, "
        f"Q={summary['min_heat_rejection_w']:.2f}.."
        f"{summary['max_heat_rejection_w']:.2f} W, "
        f"air={summary['min_target_air_flow_slpm']:.1f}.."
        f"{summary['max_target_air_flow_slpm']:.1f} slpm",
        flush=True,
    )
    print(
        f"mean air purge={summary['mean_purge_target_air_flow_slpm']} "
        f"nonpurge={summary['mean_nonpurge_target_air_flow_slpm']}",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
