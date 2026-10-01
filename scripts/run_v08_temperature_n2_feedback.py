"""Compare fixed-T V0.7 dynamics with prescribed V0.8 thermal feedback.

This targeted V0.8 step applies the validated lumped temperature trajectory to:
- gas-law and water-vapour thermodynamics;
- humid cathode N2 partial pressure;
- direct Catalano Arrhenius scaling of N2 permeance.

The Motupally membrane lookup remains at the V0.7 baseline temperature here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.anode_nitrogen import humid_air_nitrogen_partial_pressure_pa
from pemfc_dedalus.gas_permeability import (
    arrhenius_permeability,
    barrer_to_si_permeability,
)
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
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


def relative_change(new: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("reference must be non-zero")
    return new / reference - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure-csv",
        type=Path,
        default=Path("results/v06-rha-sensitivity.csv"),
    )
    parser.add_argument(
        "--temperature-csv",
        type=Path,
        default=Path("results/quick-v08-lumped-thermal-dynamics.csv"),
    )
    parser.add_argument("--regime", default="nominal")
    parser.add_argument("--feedback-exponent", type=float, default=1.0)
    parser.add_argument("--stop-time", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--lookup-rh-points", type=int, default=21)
    parser.add_argument("--lookup-current-points", type=int, default=21)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v08-temperature-n2-feedback.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v08-temperature-n2-feedback.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
    if args.regime not in closure:
        parser.error(f"unknown regime: {args.regime}")

    trajectory_time, trajectory_temperature = read_temperature_trajectory(
        args.temperature_csv
    )

    def temperature_model(time_s: float) -> float:
        return float(
            np.interp(
                time_s,
                trajectory_time,
                trajectory_temperature,
                left=trajectory_temperature[0],
                right=trajectory_temperature[-1],
            )
        )

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
        float(patch_current_a)
        * scaling.area_scale_factor
        / scaling.inferred_active_area_m2
        for patch_current_a in closure[args.regime][2]
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
    baseline_permeance_model = make_tabulated_permeance_model(
        rh_axis,
        current_axis,
        lookup,
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
        temperature_k: float,
    ) -> float:
        baseline = baseline_permeance_model(
            relative_humidity,
            current_density_a_m2,
        )
        dry_at_temperature = arrhenius_permeability(
            dry_reference_si,
            temperature_k,
            REFERENCE_TEMPERATURE_K,
            DRY_N2_ACTIVATION_ENERGY_J_MOL,
            p.gas_constant,
        )
        return baseline * dry_at_temperature / baseline_dry_permeability

    def cathode_n2_pressure_model(temperature_k: float) -> float:
        return humid_air_nitrogen_partial_pressure_pa(
            total_pressure_pa=p.pressure,
            oxygen_dry_mole_fraction=p.oxygen_mole_fraction,
            relative_humidity=p.relative_humidity,
            saturation_water_pressure_pa=water_saturation_pressure_pa(
                temperature_k
            ),
        )

    baseline_cathode_n2 = cathode_n2_pressure_model(p.stack_temperature)

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
    }

    _, baseline_summary = simulate_nitrogen_regime(
        args.regime,
        closure[args.regime],
        n2_state_permeance_model=baseline_permeance_model,
        cathode_n2_partial_pressure_pa=baseline_cathode_n2,
        **common,
    )

    _, thermal_summary = simulate_nitrogen_regime(
        args.regime,
        closure[args.regime],
        n2_state_temperature_permeance_model=thermal_permeance_model,
        cathode_n2_partial_pressure_pa=baseline_cathode_n2,
        cathode_n2_partial_pressure_model=cathode_n2_pressure_model,
        temperature_model_k=temperature_model,
        **common,
    )

    keys = [
        "mean_n2_crossover_flux_mol_m2_s",
        "max_nitrogen_mole_fraction",
        "mean_cell_current_a",
        "mean_purge_period_s",
        "cumulative_n2_crossover_mol",
        "cumulative_n2_purged_mol",
        "final_relative_humidity",
    ]
    rows: list[dict[str, Any]] = []
    for key in keys:
        baseline_value = float(baseline_summary[key])
        thermal_value = float(thermal_summary[key])
        rows.append(
            {
                "metric": key,
                "fixed_temperature": baseline_value,
                "prescribed_temperature": thermal_value,
                "relative_change": relative_change(
                    thermal_value,
                    baseline_value,
                ),
            }
        )

    output = {
        "schema_version": 1,
        "model": "v08-temperature-n2-feedback",
        "regime": args.regime,
        "feedback_exponent": args.feedback_exponent,
        "temperature_source": str(args.temperature_csv),
        "baseline_temperature_k": p.stack_temperature,
        "temperature_initial_k": temperature_model(0.0),
        "temperature_final_k": temperature_model(args.stop_time),
        "baseline_summary": baseline_summary,
        "thermal_summary": thermal_summary,
        "comparison": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['metric']}: "
            f"{row['fixed_temperature']:.6g} -> "
            f"{row['prescribed_temperature']:.6g} "
            f"({100.0 * row['relative_change']:+.2f}%)",
            flush=True,
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
