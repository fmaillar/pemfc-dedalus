"""Compare constant-k_a and Grimaldi transfer with Motupally membrane transport."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.anode_nitrogen import humid_air_nitrogen_partial_pressure_pa
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime
from scripts.run_v07_n2_catalano_membrane_transport import (
    DEFAULT_ACTIVATION_ENERGY_J_MOL,
    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
)
from scripts.run_v07_n2_catalano_motupally import (
    compute_motupally_lookup_row,
    make_tabulated_permeance_model,
)

DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty Grimaldi comparison")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_lookup_task(
    task: dict[str, Any],
) -> tuple[str, float, list[float]]:
    relative_humidity, values = compute_motupally_lookup_row(task)
    return str(task["transfer_model"]), relative_humidity, values


def build_lookup_tables(
    *,
    relative_humidity_axis: np.ndarray,
    current_density_axis: np.ndarray,
    jobs: int,
    activation_energy_j_mol: float,
    water_partial_molar_volume_cm3_mol: float,
    constant_transfer_coefficient_m_s: float,
) -> dict[str, np.ndarray]:
    tasks: list[dict[str, Any]] = []
    for transfer_model in ("constant", "grimaldi"):
        for relative_humidity in relative_humidity_axis:
            tasks.append(
                {
                    "transfer_model": transfer_model,
                    "relative_humidity": float(relative_humidity),
                    "current_density_axis": [
                        float(value) for value in current_density_axis
                    ],
                    "activation_energy_j_mol": activation_energy_j_mol,
                    "water_partial_molar_volume_cm3_mol": (
                        water_partial_molar_volume_cm3_mol
                    ),
                    "anode_transfer_coefficient_m_s": (
                        constant_transfer_coefficient_m_s
                    ),
                }
            )

    worker_count = min(jobs, len(tasks))
    if worker_count == 1:
        results = [run_lookup_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(run_lookup_task, tasks))

    grouped: dict[str, list[tuple[float, list[float]]]] = {
        "constant": [],
        "grimaldi": [],
    }
    for transfer_model, relative_humidity, values in results:
        grouped[transfer_model].append((relative_humidity, values))

    tables: dict[str, np.ndarray] = {}
    for transfer_model, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        tables[transfer_model] = np.asarray(
            [values for _, values in rows],
            dtype=float,
        )
    return tables


def run_case(task: dict[str, Any]) -> dict[str, Any]:
    permeance_model = make_tabulated_permeance_model(
        np.asarray(task["lookup_rh_axis"], dtype=float),
        np.asarray(task["lookup_current_axis"], dtype=float),
        np.asarray(task["lookup_table"], dtype=float),
    )
    _, summary = simulate_nitrogen_regime(
        str(task["regime"]),
        task["closure"],
        hydrogen_feedback_exponent=float(task["feedback_exponent"]),
        n2_state_permeance_model=permeance_model,
        **task["common_kwargs"],
    )
    return {
        "regime": str(task["regime"]),
        "hydrogen_feedback_exponent": float(task["feedback_exponent"]),
        "transfer_model": str(task["transfer_model"]),
        "summary": summary,
    }


def relative_change(new: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("reference value must be non-zero")
    return new / reference - 1.0


def combine_results(
    simulation_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for result in simulation_results:
        key = (
            str(result["regime"]),
            float(result["hydrogen_feedback_exponent"]),
        )
        grouped.setdefault(key, {})[str(result["transfer_model"])] = (
            result["summary"]
        )

    rows: list[dict[str, Any]] = []
    for regime, gamma in sorted(grouped, key=lambda item: (item[1], item[0])):
        pair = grouped[(regime, gamma)]
        constant = pair["constant"]
        grimaldi = pair["grimaldi"]
        constant_flux = float(constant["mean_n2_crossover_flux_mol_m2_s"])
        grimaldi_flux = float(grimaldi["mean_n2_crossover_flux_mol_m2_s"])
        constant_xn2 = float(constant["max_nitrogen_mole_fraction"])
        grimaldi_xn2 = float(grimaldi["max_nitrogen_mole_fraction"])

        rows.append(
            {
                "regime": regime,
                "hydrogen_feedback_exponent": gamma,
                "constant_ka_jmean_mol_m2_s": constant_flux,
                "grimaldi_jmean_mol_m2_s": grimaldi_flux,
                "jmean_change_fraction": relative_change(
                    grimaldi_flux,
                    constant_flux,
                ),
                "constant_ka_xn2_max": constant_xn2,
                "grimaldi_xn2_max": grimaldi_xn2,
                "xn2_change_fraction": relative_change(
                    grimaldi_xn2,
                    constant_xn2,
                ),
                "constant_ka_mean_cell_current_a": (
                    constant["mean_cell_current_a"]
                ),
                "grimaldi_mean_cell_current_a": (
                    grimaldi["mean_cell_current_a"]
                ),
                "constant_ka_purge_count": constant["purge_count"],
                "grimaldi_purge_count": grimaldi["purge_count"],
                "constant_ka_mean_purge_period_s": (
                    constant["mean_purge_period_s"]
                ),
                "grimaldi_mean_purge_period_s": (
                    grimaldi["mean_purge_period_s"]
                ),
                "constant_ka_h2_balance_error_mol": (
                    constant["h2_balance_error_mol"]
                ),
                "grimaldi_h2_balance_error_mol": (
                    grimaldi["h2_balance_error_mol"]
                ),
                "constant_ka_n2_balance_error_mol": (
                    constant["n2_balance_error_mol"]
                ),
                "grimaldi_n2_balance_error_mol": (
                    grimaldi["n2_balance_error_mol"]
                ),
                "constant_ka_water_balance_error_mol": (
                    constant["water_balance_error_mol"]
                ),
                "grimaldi_water_balance_error_mol": (
                    grimaldi["water_balance_error_mol"]
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
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--lookup-rh-points", type=int, default=33)
    parser.add_argument("--lookup-current-points", type=int, default=33)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-n2-grimaldi-transfer-comparison.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-n2-grimaldi-transfer-comparison.csv"),
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.lookup_rh_points < 3 or args.lookup_current_points < 3:
        parser.error("lookup axes must each contain at least three points")

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

    current_density_values: list[float] = []
    for regime in args.regimes:
        for patch_current_a in closure[regime][2]:
            cell_current_a = float(patch_current_a) * scaling.area_scale_factor
            current_density_values.append(
                cell_current_a / scaling.inferred_active_area_m2
            )
    current_min = min(current_density_values)
    current_max = max(current_density_values)
    margin = max(0.1 * (current_max - current_min), 50.0)
    rh_axis = np.linspace(0.0, 1.0, args.lookup_rh_points)
    current_axis = np.linspace(
        max(0.0, current_min - margin),
        current_max + margin,
        args.lookup_current_points,
    )

    lookup_task_count = 2 * args.lookup_rh_points
    print(
        "Precomputing constant-k_a and Grimaldi lookups: "
        f"{lookup_task_count} RH/model tasks on "
        f"{min(args.jobs, lookup_task_count)} workers",
        flush=True,
    )
    lookup_tables = build_lookup_tables(
        relative_humidity_axis=rh_axis,
        current_density_axis=current_axis,
        jobs=args.jobs,
        activation_energy_j_mol=args.activation_energy_j_mol,
        water_partial_molar_volume_cm3_mol=(
            args.water_partial_molar_volume_cm3_mol
        ),
        constant_transfer_coefficient_m_s=(
            p.anode_water_transfer_coefficient
        ),
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
            for transfer_model in ("constant", "grimaldi"):
                tasks.append(
                    {
                        "regime": regime,
                        "feedback_exponent": gamma,
                        "transfer_model": transfer_model,
                        "closure": closure[regime],
                        "common_kwargs": common_kwargs,
                        "lookup_rh_axis": [float(value) for value in rh_axis],
                        "lookup_current_axis": [
                            float(value) for value in current_axis
                        ],
                        "lookup_table": (
                            lookup_tables[transfer_model].tolist()
                        ),
                    }
                )

    worker_count = min(args.jobs, len(tasks))
    if worker_count == 1:
        results = [run_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(run_case, tasks))

    rows = combine_results(results)
    for row in rows:
        print(
            f"{row['regime']} gamma={row['hydrogen_feedback_exponent']:.1f}: "
            f"J {row['constant_ka_jmean_mol_m2_s']:.3e}->"
            f"{row['grimaldi_jmean_mol_m2_s']:.3e} "
            f"({100.0 * row['jmean_change_fraction']:+.1f}%), "
            f"xN2,max {row['constant_ka_xn2_max']:.5f}->"
            f"{row['grimaldi_xn2_max']:.5f} "
            f"({100.0 * row['xn2_change_fraction']:+.1f}%)",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-n2-grimaldi-transfer-comparison",
        "comparison": "constant_k_a_vs_grimaldi_k_g",
        "constant_k_a_m_s": p.anode_water_transfer_coefficient,
        "regimes": args.regimes,
        "feedback_exponents": args.feedback_exponents,
        "jobs": worker_count,
        "lookup_rh_points": args.lookup_rh_points,
        "lookup_current_points": args.lookup_current_points,
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
