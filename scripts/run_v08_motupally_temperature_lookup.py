"""Compare Arrhenius-only and full RH x j x T Motupally thermal feedback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

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
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    temperatures = np.asarray(
        [float(row["stack_temperature_c"]) + 273.15 for row in rows],
        dtype=float,
    )
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("temperature trajectory must contain increasing times")
    return times, temperatures


def make_trilinear_permeance_model(
    temperature_axis_k: np.ndarray,
    relative_humidity_axis: np.ndarray,
    current_density_axis: np.ndarray,
    lookup_tables: np.ndarray,
) -> Callable[[float, float, float], float]:
    """Interpolate bilinearly in RH/current and linearly in temperature."""
    temperature_axis = np.asarray(temperature_axis_k, dtype=float)
    values = np.asarray(lookup_tables, dtype=float)
    if temperature_axis.ndim != 1 or temperature_axis.size < 2:
        raise ValueError("temperature axis must contain at least two points")
    if np.any(np.diff(temperature_axis) <= 0.0):
        raise ValueError("temperature axis must increase")
    if values.shape != (
        temperature_axis.size,
        len(relative_humidity_axis),
        len(current_density_axis),
    ):
        raise ValueError("lookup volume shape does not match axes")

    plane_models = [
        make_tabulated_permeance_model(
            relative_humidity_axis,
            current_density_axis,
            values[index],
        )
        for index in range(temperature_axis.size)
    ]

    def interpolate(
        relative_humidity: float,
        current_density_a_m2: float,
        temperature_k: float,
    ) -> float:
        temperature = float(
            np.clip(
                temperature_k,
                temperature_axis[0],
                temperature_axis[-1],
            )
        )
        upper = int(np.searchsorted(temperature_axis, temperature, side="right"))
        upper = min(max(upper, 1), temperature_axis.size - 1)
        lower = upper - 1
        t0 = float(temperature_axis[lower])
        t1 = float(temperature_axis[upper])
        weight = 0.0 if t1 == t0 else (temperature - t0) / (t1 - t0)
        v0 = plane_models[lower](
            relative_humidity,
            current_density_a_m2,
        )
        v1 = plane_models[upper](
            relative_humidity,
            current_density_a_m2,
        )
        return float((1.0 - weight) * v0 + weight * v1)

    return interpolate


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
    parser.add_argument("--lookup-rh-points", type=int, default=11)
    parser.add_argument("--lookup-current-points", type=int, default=11)
    parser.add_argument("--lookup-temperature-points", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v08-motupally-temperature-lookup.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v08-motupally-temperature-lookup.csv"),
    )
    args = parser.parse_args()

    if args.lookup_temperature_points < 2:
        parser.error("--lookup-temperature-points must be >= 2")

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
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
    temperature_axis = np.linspace(
        min(float(np.min(trajectory_temperature)), p.stack_temperature),
        max(float(np.max(trajectory_temperature)), p.stack_temperature),
        args.lookup_temperature_points,
    )

    print(
        "Precomputing Motupally temperature lookup: "
        f"{args.lookup_temperature_points} x "
        f"{args.lookup_rh_points} x {args.lookup_current_points}",
        flush=True,
    )
    tables = []
    for temperature_k in temperature_axis:
        print(f"  T={temperature_k:.2f} K", flush=True)
        tables.append(
            build_motupally_lookup_table(
                relative_humidity_axis=rh_axis,
                current_density_axis=current_axis,
                jobs=args.jobs,
                activation_energy_j_mol=DEFAULT_ACTIVATION_ENERGY_J_MOL,
                water_partial_molar_volume_cm3_mol=(
                    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL
                ),
                anode_transfer_coefficient_m_s=(
                    p.anode_water_transfer_coefficient
                ),
                stack_temperature_k=float(temperature_k),
            )
        )
    lookup_volume = np.asarray(tables, dtype=float)

    full_temperature_model = make_trilinear_permeance_model(
        temperature_axis,
        rh_axis,
        current_axis,
        lookup_volume,
    )

    baseline_table = build_motupally_lookup_table(
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
    baseline_model = make_tabulated_permeance_model(
        rh_axis,
        current_axis,
        baseline_table,
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

    def arrhenius_only_model(
        relative_humidity: float,
        current_density_a_m2: float,
        temperature_k: float,
    ) -> float:
        dry_at_temperature = arrhenius_permeability(
            dry_reference_si,
            temperature_k,
            REFERENCE_TEMPERATURE_K,
            DRY_N2_ACTIVATION_ENERGY_J_MOL,
            p.gas_constant,
        )
        return (
            baseline_model(relative_humidity, current_density_a_m2)
            * dry_at_temperature
            / baseline_dry_permeability
        )

    def cathode_n2_pressure_model(temperature_k: float) -> float:
        return humid_air_nitrogen_partial_pressure_pa(
            total_pressure_pa=p.pressure,
            oxygen_dry_mole_fraction=p.oxygen_mole_fraction,
            relative_humidity=p.relative_humidity,
            saturation_water_pressure_pa=water_saturation_pressure_pa(
                temperature_k
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
        "cathode_n2_partial_pressure_model": cathode_n2_pressure_model,
        "temperature_model_k": temperature_model,
    }

    summaries: dict[str, dict[str, Any]] = {}
    for name, model in (
        ("arrhenius_only", arrhenius_only_model),
        ("full_temperature_lookup", full_temperature_model),
    ):
        _, summary = simulate_nitrogen_regime(
            args.regime,
            closure[args.regime],
            n2_state_temperature_permeance_model=model,
            cathode_n2_partial_pressure_pa=cathode_n2_pressure_model(
                p.stack_temperature
            ),
            **common,
        )
        summaries[name] = summary

    metrics = [
        "mean_n2_crossover_flux_mol_m2_s",
        "max_nitrogen_mole_fraction",
        "mean_cell_current_a",
        "mean_purge_period_s",
        "cumulative_n2_crossover_mol",
        "cumulative_n2_purged_mol",
        "final_relative_humidity",
    ]
    comparison: list[dict[str, Any]] = []
    for metric in metrics:
        reference = float(summaries["arrhenius_only"][metric])
        full = float(summaries["full_temperature_lookup"][metric])
        comparison.append(
            {
                "metric": metric,
                "arrhenius_only": reference,
                "full_temperature_lookup": full,
                "relative_change": relative_change(full, reference),
            }
        )

    output = {
        "schema_version": 1,
        "model": "v08-motupally-temperature-lookup",
        "temperature_axis_k": temperature_axis.tolist(),
        "lookup_shape": list(lookup_volume.shape),
        "lookup_rh_points": args.lookup_rh_points,
        "lookup_current_points": args.lookup_current_points,
        "lookup_temperature_points": args.lookup_temperature_points,
        "summaries": summaries,
        "comparison": comparison,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    for row in comparison:
        print(
            f"{row['metric']}: "
            f"{row['arrhenius_only']:.6g} -> "
            f"{row['full_temperature_lookup']:.6g} "
            f"({100.0 * row['relative_change']:+.2f}%)",
            flush=True,
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
