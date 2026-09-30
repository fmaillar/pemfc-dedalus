"""Compare constant-D and Motupally variable-D membrane profiles for N2 crossover."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.anode_nitrogen import humid_air_nitrogen_partial_pressure_pa
from pemfc_dedalus.gas_permeability import (
    arrhenius_permeability,
    barrer_to_si_permeability,
    calibrated_exponential_permeability_multiplier,
    membrane_permeance_from_permeability_profile,
    membrane_water_volume_fraction_from_partial_molar_volume,
)
from pemfc_dedalus.membrane import (
    electro_osmotic_lambda_velocity,
    membrane_fixed_charge_concentration,
    membrane_water_content_from_activity,
    steady_membrane_water_profile_motupally,
)
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import REGIMES, interpolate_flux_and_current, load_h2_closure
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime
from scripts.run_v07_n2_catalano_calibrated import (
    ANCHOR_PERMEABILITY_FACTOR,
    ANCHOR_WATER_ACTIVITY,
    DRY_REFERENCE_PERMEABILITY_BARRER,
    MAXIMUM_PERMEABILITY_FACTOR,
    MEMBRANE_PROFILE_POINTS,
    REFERENCE_TEMPERATURE_K,
)
from scripts.run_v07_n2_catalano_membrane_transport import (
    DEFAULT_ACTIVATION_ENERGY_J_MOL,
    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
    make_v06_transport_permeance_model,
)

DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]


def scalar_water_content(activity: float) -> float:
    return float(membrane_water_content_from_activity(activity))


def make_motupally_transport_permeance_model(
    *,
    dry_reference_si: float,
    stack_temperature_k: float,
    gas_constant_j_mol_k: float,
    faraday_c_mol: float,
    fixed_charge_mol_m3: float,
    activation_energy_j_mol: float,
    cathode_relative_humidity: float,
    membrane_thickness_m: float,
    membrane_equivalent_weight_kg_mol: float,
    membrane_dry_density_kg_m3: float,
    anode_transfer_coefficient_m_s: float,
    water_partial_molar_volume_m3_mol: float,
    profile_points: int = MEMBRANE_PROFILE_POINTS,
) -> Callable[[float, float], float]:
    """Build Catalano permeance on a Motupally variable-D lambda profile."""
    if profile_points < 2:
        raise ValueError("profile_points must be >= 2")

    lambda_dry = scalar_water_content(0.0)
    lambda_anchor = scalar_water_content(ANCHOR_WATER_ACTIVITY)
    lambda_cathode = scalar_water_content(cathode_relative_humidity)
    phi_dry = membrane_water_volume_fraction_from_partial_molar_volume(
        lambda_dry,
        membrane_equivalent_weight_kg_mol=membrane_equivalent_weight_kg_mol,
        membrane_dry_density_kg_m3=membrane_dry_density_kg_m3,
        water_partial_molar_volume_m3_mol=water_partial_molar_volume_m3_mol,
    )
    phi_anchor = membrane_water_volume_fraction_from_partial_molar_volume(
        lambda_anchor,
        membrane_equivalent_weight_kg_mol=membrane_equivalent_weight_kg_mol,
        membrane_dry_density_kg_m3=membrane_dry_density_kg_m3,
        water_partial_molar_volume_m3_mol=water_partial_molar_volume_m3_mol,
    )
    dry_at_temperature = arrhenius_permeability(
        dry_reference_si,
        stack_temperature_k,
        REFERENCE_TEMPERATURE_K,
        activation_energy_j_mol,
        gas_constant_j_mol_k,
    )
    z_membrane = np.linspace(0.0, membrane_thickness_m, profile_points)

    def permeance_model(
        anode_relative_humidity: float,
        current_density_a_m2: float,
    ) -> float:
        lambda_anode_equilibrium = scalar_water_content(anode_relative_humidity)
        drag_velocity = electro_osmotic_lambda_velocity(
            current_density_a_m2,
            faraday_c_mol,
            fixed_charge_mol_m3,
        )
        lambda_profile = steady_membrane_water_profile_motupally(
            z_membrane,
            lambda_cathode=lambda_cathode,
            lambda_anode_equilibrium=lambda_anode_equilibrium,
            temperature_k=stack_temperature_k,
            drag_velocity_m_s=drag_velocity,
            anode_transfer_coefficient_m_s=anode_transfer_coefficient_m_s,
        )

        permeability_profile: list[float] = []
        for lambda_local in lambda_profile:
            phi_water = membrane_water_volume_fraction_from_partial_molar_volume(
                float(lambda_local),
                membrane_equivalent_weight_kg_mol=(
                    membrane_equivalent_weight_kg_mol
                ),
                membrane_dry_density_kg_m3=membrane_dry_density_kg_m3,
                water_partial_molar_volume_m3_mol=(
                    water_partial_molar_volume_m3_mol
                ),
            )
            humidity_factor = calibrated_exponential_permeability_multiplier(
                phi_water,
                baseline_fraction=phi_dry,
                anchor_fraction=phi_anchor,
                anchor_factor=ANCHOR_PERMEABILITY_FACTOR,
                maximum_factor=MAXIMUM_PERMEABILITY_FACTOR,
            )
            permeability_profile.append(
                dry_at_temperature * humidity_factor
            )

        return membrane_permeance_from_permeability_profile(
            permeability_profile,
            membrane_thickness_m,
        )

    return permeance_model



def compute_motupally_lookup_row(task: dict[str, Any]) -> tuple[float, list[float]]:
    """Compute one RH row of the Motupally permeance lookup table."""
    p = CathodeParameters()
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    model = make_motupally_transport_permeance_model(
        dry_reference_si=barrer_to_si_permeability(
            DRY_REFERENCE_PERMEABILITY_BARRER
        ),
        stack_temperature_k=p.stack_temperature,
        gas_constant_j_mol_k=p.gas_constant,
        faraday_c_mol=p.faraday,
        fixed_charge_mol_m3=fixed_charge,
        activation_energy_j_mol=float(task["activation_energy_j_mol"]),
        cathode_relative_humidity=p.relative_humidity,
        membrane_thickness_m=p.membrane_thickness,
        membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        membrane_dry_density_kg_m3=p.membrane_dry_density,
        anode_transfer_coefficient_m_s=float(
            task["anode_transfer_coefficient_m_s"]
        ),
        water_partial_molar_volume_m3_mol=(
            float(task["water_partial_molar_volume_cm3_mol"]) * 1.0e-6
        ),
    )
    relative_humidity = float(task["relative_humidity"])
    values = [
        model(relative_humidity, float(current_density))
        for current_density in task["current_density_axis"]
    ]
    return relative_humidity, values


def build_motupally_lookup_table(
    *,
    relative_humidity_axis: np.ndarray,
    current_density_axis: np.ndarray,
    jobs: int,
    activation_energy_j_mol: float,
    water_partial_molar_volume_cm3_mol: float,
    anode_transfer_coefficient_m_s: float,
) -> np.ndarray:
    """Precompute Motupally permeance on a 2D state grid in parallel."""
    tasks = [
        {
            "relative_humidity": float(relative_humidity),
            "current_density_axis": [
                float(value) for value in current_density_axis
            ],
            "activation_energy_j_mol": activation_energy_j_mol,
            "water_partial_molar_volume_cm3_mol": (
                water_partial_molar_volume_cm3_mol
            ),
            "anode_transfer_coefficient_m_s": (
                anode_transfer_coefficient_m_s
            ),
        }
        for relative_humidity in relative_humidity_axis
    ]
    worker_count = min(jobs, len(tasks))
    if worker_count == 1:
        rows = [compute_motupally_lookup_row(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(compute_motupally_lookup_row, tasks))
    rows.sort(key=lambda item: item[0])
    return np.asarray([row[1] for row in rows], dtype=float)


def make_tabulated_permeance_model(
    relative_humidity_axis: np.ndarray,
    current_density_axis: np.ndarray,
    table: np.ndarray,
) -> Callable[[float, float], float]:
    """Return bilinear interpolation over a precomputed permeance table."""
    rh_axis = np.asarray(relative_humidity_axis, dtype=float)
    current_axis = np.asarray(current_density_axis, dtype=float)
    values = np.asarray(table, dtype=float)
    if values.shape != (rh_axis.size, current_axis.size):
        raise ValueError("lookup table shape does not match axes")
    if rh_axis.size < 2 or current_axis.size < 2:
        raise ValueError("lookup axes must each contain at least two points")

    def interpolate(relative_humidity: float, current_density_a_m2: float) -> float:
        rh = float(np.clip(relative_humidity, rh_axis[0], rh_axis[-1]))
        current = float(
            np.clip(current_density_a_m2, current_axis[0], current_axis[-1])
        )

        rh_upper = int(np.searchsorted(rh_axis, rh, side="right"))
        current_upper = int(
            np.searchsorted(current_axis, current, side="right")
        )
        rh_upper = min(max(rh_upper, 1), rh_axis.size - 1)
        current_upper = min(
            max(current_upper, 1),
            current_axis.size - 1,
        )
        rh_lower = rh_upper - 1
        current_lower = current_upper - 1

        rh0 = float(rh_axis[rh_lower])
        rh1 = float(rh_axis[rh_upper])
        current0 = float(current_axis[current_lower])
        current1 = float(current_axis[current_upper])
        rh_weight = 0.0 if rh1 == rh0 else (rh - rh0) / (rh1 - rh0)
        current_weight = (
            0.0
            if current1 == current0
            else (current - current0) / (current1 - current0)
        )

        lower_value = (
            (1.0 - current_weight) * values[rh_lower, current_lower]
            + current_weight * values[rh_lower, current_upper]
        )
        upper_value = (
            (1.0 - current_weight) * values[rh_upper, current_lower]
            + current_weight * values[rh_upper, current_upper]
        )
        return float(
            (1.0 - rh_weight) * lower_value
            + rh_weight * upper_value
        )

    return interpolate

def relative_change(new: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("reference value must be non-zero")
    return new / reference - 1.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty Motupally comparison")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_simulation_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one independent regime/gamma/model simulation."""
    p = CathodeParameters()
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    water_volume_m3_mol = (
        float(task["water_partial_molar_volume_cm3_mol"]) * 1.0e-6
    )
    dry_reference_si = barrer_to_si_permeability(
        DRY_REFERENCE_PERMEABILITY_BARRER
    )
    activation_energy = float(task["activation_energy_j_mol"])
    model_kind = str(task["model_kind"])

    if model_kind == "constant_d":
        permeance_model = make_v06_transport_permeance_model(
            dry_reference_si=dry_reference_si,
            stack_temperature_k=p.stack_temperature,
            gas_constant_j_mol_k=p.gas_constant,
            faraday_c_mol=p.faraday,
            fixed_charge_mol_m3=fixed_charge,
            activation_energy_j_mol=activation_energy,
            cathode_relative_humidity=p.relative_humidity,
            membrane_thickness_m=p.membrane_thickness,
            membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
            membrane_dry_density_kg_m3=p.membrane_dry_density,
            membrane_water_diffusivity_m2_s=p.membrane_water_diffusivity,
            anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
            water_partial_molar_volume_m3_mol=water_volume_m3_mol,
        )
    elif model_kind == "motupally":
        permeance_model = make_tabulated_permeance_model(
            np.asarray(task["lookup_rh_axis"], dtype=float),
            np.asarray(task["lookup_current_axis"], dtype=float),
            np.asarray(task["lookup_table"], dtype=float),
        )
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")

    regime = str(task["regime"])
    gamma = float(task["feedback_exponent"])
    _, summary = simulate_nitrogen_regime(
        regime,
        task["closure"],
        hydrogen_feedback_exponent=gamma,
        n2_state_permeance_model=permeance_model,
        **task["common_kwargs"],
    )
    return {
        "regime": regime,
        "hydrogen_feedback_exponent": gamma,
        "model_kind": model_kind,
        "summary": summary,
    }


def combine_model_results(
    simulation_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair constant-D and Motupally results for each regime/gamma case."""
    grouped: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for result in simulation_results:
        key = (
            str(result["regime"]),
            float(result["hydrogen_feedback_exponent"]),
        )
        grouped.setdefault(key, {})[str(result["model_kind"])] = result["summary"]

    rows: list[dict[str, Any]] = []
    for regime, gamma in sorted(grouped, key=lambda item: (item[1], item[0])):
        pair = grouped[(regime, gamma)]
        constant_summary = pair["constant_d"]
        motupally_summary = pair["motupally"]

        constant_flux = float(
            constant_summary["mean_n2_crossover_flux_mol_m2_s"]
        )
        motupally_flux = float(
            motupally_summary["mean_n2_crossover_flux_mol_m2_s"]
        )
        constant_xn2 = float(constant_summary["max_nitrogen_mole_fraction"])
        motupally_xn2 = float(motupally_summary["max_nitrogen_mole_fraction"])

        rows.append(
            {
                "regime": regime,
                "hydrogen_feedback_exponent": gamma,
                "constant_d_jmean_mol_m2_s": constant_flux,
                "motupally_jmean_mol_m2_s": motupally_flux,
                "jmean_change_fraction": relative_change(
                    motupally_flux,
                    constant_flux,
                ),
                "constant_d_xn2_max": constant_xn2,
                "motupally_xn2_max": motupally_xn2,
                "xn2_change_fraction": relative_change(
                    motupally_xn2,
                    constant_xn2,
                ),
                "constant_d_mean_cell_current_a": (
                    constant_summary["mean_cell_current_a"]
                ),
                "motupally_mean_cell_current_a": (
                    motupally_summary["mean_cell_current_a"]
                ),
                "constant_d_purge_count": constant_summary["purge_count"],
                "motupally_purge_count": motupally_summary["purge_count"],
                "constant_d_mean_purge_period_s": (
                    constant_summary["mean_purge_period_s"]
                ),
                "motupally_mean_purge_period_s": (
                    motupally_summary["mean_purge_period_s"]
                ),
                "constant_d_h2_balance_error_mol": (
                    constant_summary["h2_balance_error_mol"]
                ),
                "motupally_h2_balance_error_mol": (
                    motupally_summary["h2_balance_error_mol"]
                ),
                "constant_d_n2_balance_error_mol": (
                    constant_summary["n2_balance_error_mol"]
                ),
                "motupally_n2_balance_error_mol": (
                    motupally_summary["n2_balance_error_mol"]
                ),
                "constant_d_water_balance_error_mol": (
                    constant_summary["water_balance_error_mol"]
                ),
                "motupally_water_balance_error_mol": (
                    motupally_summary["water_balance_error_mol"]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure-csv",
        type=Path,
        default=Path("results/v06-rha-sensitivity.csv"),
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=list(REGIMES),
        default=list(REGIMES),
    )
    parser.add_argument(
        "--feedback-exponents",
        nargs="+",
        type=float,
        default=DEFAULT_FEEDBACK_EXPONENTS,
    )
    parser.add_argument(
        "--water-partial-molar-volume-cm3-mol",
        type=float,
        default=DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
    )
    parser.add_argument(
        "--activation-energy-j-mol",
        type=float,
        default=DEFAULT_ACTIVATION_ENERGY_J_MOL,
    )
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--lookup-rh-points", type=int, default=33)
    parser.add_argument("--lookup-current-points", type=int, default=33)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-n2-catalano-motupally-comparison.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-n2-catalano-motupally-comparison.csv"),
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.lookup_rh_points < 3 or args.lookup_current_points < 3:
        parser.error("lookup axes must contain at least three points")
    if args.stop_time <= 0.0 or args.dt <= 0.0:
        parser.error("--stop-time and --dt must be positive")

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
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
    selected_current_density_values: list[float] = []
    for regime in args.regimes:
        current_values = closure[regime][2]
        for patch_current_a in current_values:
            cell_current_a = float(patch_current_a) * scaling.area_scale_factor
            selected_current_density_values.append(
                cell_current_a / scaling.inferred_active_area_m2
            )
    min_current_density = min(selected_current_density_values)
    max_current_density = max(selected_current_density_values)
    current_margin = max(
        0.1 * (max_current_density - min_current_density),
        50.0,
    )
    lookup_rh_axis = np.linspace(0.0, 1.0, args.lookup_rh_points)
    lookup_current_axis = np.linspace(
        max(0.0, min_current_density - current_margin),
        max_current_density + current_margin,
        args.lookup_current_points,
    )
    print(
        "Precomputing Motupally lookup: "
        f"{args.lookup_rh_points}x{args.lookup_current_points} "
        f"with {min(args.jobs, args.lookup_rh_points)} workers",
        flush=True,
    )
    lookup_table = build_motupally_lookup_table(
        relative_humidity_axis=lookup_rh_axis,
        current_density_axis=lookup_current_axis,
        jobs=args.jobs,
        activation_energy_j_mol=args.activation_energy_j_mol,
        water_partial_molar_volume_cm3_mol=(
            args.water_partial_molar_volume_cm3_mol
        ),
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )

    common_kwargs: dict[str, Any] = {
        "initial_rh": args.initial_rh,
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
        "cathode_n2_partial_pressure_pa": cathode_n2_partial_pressure,
    }

    tasks: list[dict[str, Any]] = []
    for gamma in args.feedback_exponents:
        for regime in args.regimes:
            for model_kind in ("constant_d", "motupally"):
                tasks.append(
                    {
                        "regime": regime,
                        "feedback_exponent": gamma,
                        "model_kind": model_kind,
                        "closure": closure[regime],
                        "common_kwargs": common_kwargs,
                        "water_partial_molar_volume_cm3_mol": (
                            args.water_partial_molar_volume_cm3_mol
                        ),
                        "activation_energy_j_mol": args.activation_energy_j_mol,
                        "lookup_rh_axis": [
                            float(value) for value in lookup_rh_axis
                        ],
                        "lookup_current_axis": [
                            float(value) for value in lookup_current_axis
                        ],
                        "lookup_table": lookup_table.tolist(),
                    }
                )

    worker_count = min(args.jobs, len(tasks))
    if worker_count == 1:
        simulation_results = [run_simulation_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            simulation_results = list(
                executor.map(run_simulation_task, tasks)
            )
    rows = combine_model_results(simulation_results)

    for row in rows:
        print(
            f"{row['regime']} gamma={row['hydrogen_feedback_exponent']:.1f}: "
            f"J {row['constant_d_jmean_mol_m2_s']:.3e}->"
            f"{row['motupally_jmean_mol_m2_s']:.3e} "
            f"({100.0 * row['jmean_change_fraction']:+.1f}%), "
            f"xN2,max {row['constant_d_xn2_max']:.5f}->"
            f"{row['motupally_xn2_max']:.5f} "
            f"({100.0 * row['xn2_change_fraction']:+.1f}%)",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-n2-catalano-motupally-comparison",
        "comparison": "constant_diffusivity_vs_motupally_diffusivity",
        "closure_source": str(args.closure_csv),
        "feedback_exponents": args.feedback_exponents,
        "regimes": args.regimes,
        "jobs": worker_count,
        "lookup_rh_points": args.lookup_rh_points,
        "lookup_current_points": args.lookup_current_points,
        "lookup_current_density_min_a_m2": float(lookup_current_axis[0]),
        "lookup_current_density_max_a_m2": float(lookup_current_axis[-1]),
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "summaries": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
