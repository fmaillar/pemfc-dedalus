"""Map the validity domain of the hybrid Motupally + Grimaldi closure."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.gas_permeability import barrer_to_si_permeability
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)
from scripts.run_v07_n2_catalano_calibrated import (
    DRY_REFERENCE_PERMEABILITY_BARRER,
)
from scripts.run_v07_n2_catalano_membrane_transport import (
    DEFAULT_ACTIVATION_ENERGY_J_MOL,
    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
)
from scripts.run_v07_n2_catalano_motupally import (
    make_motupally_grimaldi_transport_permeance_model,
)


def evaluate_point(task: dict[str, Any]) -> dict[str, Any]:
    p = CathodeParameters()
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    model = make_motupally_grimaldi_transport_permeance_model(
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
        water_partial_molar_volume_m3_mol=(
            float(task["water_partial_molar_volume_cm3_mol"]) * 1.0e-6
        ),
    )

    rh = float(task["relative_humidity"])
    current_density = float(task["current_density_a_m2"])
    try:
        permeance = float(model(rh, current_density))
    except RuntimeError as exc:
        return {
            "relative_humidity": rh,
            "current_density_a_m2": current_density,
            "valid": False,
            "permeance_mol_m_m2_s_pa": "",
            "error": str(exc),
        }

    return {
        "relative_humidity": rh,
        "current_density_a_m2": current_density,
        "valid": True,
        "permeance_mol_m_m2_s_pa": permeance,
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
        "--activation-energy-j-mol",
        type=float,
        default=DEFAULT_ACTIVATION_ENERGY_J_MOL,
    )
    parser.add_argument(
        "--water-partial-molar-volume-cm3-mol",
        type=float,
        default=DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/v07-grimaldi-hybrid-validity-domain.csv"
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
            "activation_energy_j_mol": args.activation_energy_j_mol,
            "water_partial_molar_volume_cm3_mol": (
                args.water_partial_molar_volume_cm3_mol
            ),
        }
        for rh in rh_axis
        for current_density in current_axis
    ]

    worker_count = min(args.jobs, len(tasks))
    print(
        f"Evaluating {len(tasks)} Grimaldi hybrid states on "
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
    if invalid_rows:
        rh_values = [float(row["relative_humidity"]) for row in invalid_rows]
        current_values = [
            float(row["current_density_a_m2"])
            for row in invalid_rows
        ]
        print(
            "invalid domain: "
            f"RH={min(rh_values):.3f}..{max(rh_values):.3f}, "
            f"j={min(current_values):.1f}..{max(current_values):.1f} A/m2",
            flush=True,
        )
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
