"""Map validity of the coherent steady Grimaldi membrane closure."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.membrane import (
    electro_osmotic_lambda_velocity,
    membrane_fixed_charge_concentration,
    steady_membrane_water_profile_grimaldi_consistent,
)
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)


def evaluate_point(task: dict[str, Any]) -> dict[str, Any]:
    p = CathodeParameters()
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    current_density = float(task["current_density_a_m2"])
    drag_velocity = electro_osmotic_lambda_velocity(
        current_density,
        p.faraday,
        fixed_charge,
    )
    z = np.linspace(0.0, p.membrane_thickness, 129)
    rh = float(task["relative_humidity"])

    try:
        profile = steady_membrane_water_profile_grimaldi_consistent(
            z,
            anode_relative_humidity=rh,
            cathode_relative_humidity=p.relative_humidity,
            temperature_k=p.stack_temperature,
            drag_velocity_m_s=drag_velocity,
            equivalent_weight_kg_mol=p.membrane_equivalent_weight,
            dry_density_kg_m3=p.membrane_dry_density,
            gas_constant_j_mol_k=p.gas_constant,
        )
    except RuntimeError as exc:
        return {
            "relative_humidity": rh,
            "current_density_a_m2": current_density,
            "valid": False,
            "lambda_anode": "",
            "lambda_cathode": "",
            "lambda_mean": "",
            "error": str(exc),
        }

    return {
        "relative_humidity": rh,
        "current_density_a_m2": current_density,
        "valid": True,
        "lambda_anode": float(profile[0]),
        "lambda_cathode": float(profile[-1]),
        "lambda_mean": float(np.mean(profile)),
        "error": "",
    }


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
    parser.add_argument("--rh-points", type=int, default=21)
    parser.add_argument("--current-points", type=int, default=21)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/v07-grimaldi-consistent-validity-domain.csv"
        ),
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.rh_points < 3 or args.current_points < 3:
        parser.error("grid axes must each contain at least three points")

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
    rh_axis = np.linspace(0.0, 1.0, args.rh_points)
    current_axis = np.linspace(
        max(0.0, current_min - margin),
        current_max + margin,
        args.current_points,
    )

    tasks = [
        {
            "relative_humidity": float(rh),
            "current_density_a_m2": float(current_density),
        }
        for rh in rh_axis
        for current_density in current_axis
    ]
    worker_count = min(args.jobs, len(tasks))
    print(
        f"Evaluating {len(tasks)} coherent Grimaldi states on "
        f"{worker_count} workers",
        flush=True,
    )

    if worker_count == 1:
        rows = [evaluate_point(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(evaluate_point, tasks))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["valid"]]
    invalid_rows = [row for row in rows if not row["valid"]]
    print(
        f"valid={len(valid_rows)} invalid={len(invalid_rows)} "
        f"fraction_valid={len(valid_rows) / len(rows):.4f}",
        flush=True,
    )
    if valid_rows:
        lambda_anode = [float(row["lambda_anode"]) for row in valid_rows]
        lambda_cathode = [float(row["lambda_cathode"]) for row in valid_rows]
        print(
            "lambda ranges: "
            f"anode={min(lambda_anode):.3f}..{max(lambda_anode):.3f}, "
            f"cathode={min(lambda_cathode):.3f}..{max(lambda_cathode):.3f}",
            flush=True,
        )
    if invalid_rows:
        for row in invalid_rows[:10]:
            print(
                f"invalid RH={row['relative_humidity']:.3f} "
                f"j={row['current_density_a_m2']:.1f}: "
                f"{row['error']}",
                flush=True,
            )

    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
