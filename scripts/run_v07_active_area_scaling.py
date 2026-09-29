"""V0.7 full-cell active-area scaling study.

The V0.6 cathode/membrane closure is computed on a 2 mm x 1 mm numerical
patch, whereas the V0.7 anode volume and purge-flow data are per physical cell.
This driver puts those quantities on one scale.

A full-cell active area is inferred from the nominal dry-anode reference point:

    j_ref = I_patch,ref / A_patch
    A_active = I_cell,ref / j_ref

The resulting area scale is then applied consistently to:
- patch-integrated current used for H2 consumption;
- membrane area used to convert lambda-space water flux to mol/s.

The anode volume and purge flow remain per-cell quantities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)
from scripts.run_v07_anode_transient_purge import (
    simulate_transient_purge_regime,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty active-area scaling trajectory")
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
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=list(REGIMES),
        default=list(REGIMES),
    )
    parser.add_argument("--calibration-rh", type=float, default=0.0)
    parser.add_argument("--cell-reference-current", type=float, default=None)
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-active-area-scaling.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-active-area-scaling.csv"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.calibration_rh <= 1.0:
        parser.error("--calibration-rh must be in [0, 1]")
    if not 0.0 <= args.initial_rh <= 1.0:
        parser.error("--initial-rh must be in [0, 1]")
    if args.stop_time <= 0.0 or args.dt <= 0.0:
        parser.error("--stop-time and --dt must be positive")
    if args.write_every < 1:
        parser.error("--write-every must be >= 1")

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
    missing = [regime for regime in args.regimes if regime not in closure]
    if missing:
        parser.error(f"missing closure data for regimes: {', '.join(missing)}")
    if "nominal" not in closure:
        parser.error("nominal closure is required for active-area calibration")

    patch_area_m2 = p.length_x * p.length_y
    _, patch_reference_current_a = interpolate_flux_and_current(
        args.calibration_rh,
        closure["nominal"],
    )
    cell_reference_current_a = (
        p.stack_current_a
        if args.cell_reference_current is None
        else args.cell_reference_current
    )
    scaling = infer_active_area_scaling(
        patch_area_m2=patch_area_m2,
        patch_reference_current_a=patch_reference_current_a,
        cell_reference_current_a=cell_reference_current_a,
    )

    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for regime in args.regimes:
        rows, summary = simulate_transient_purge_regime(
            regime,
            closure[regime],
            initial_rh=args.initial_rh,
            stop_time_s=args.stop_time,
            dt_s=args.dt,
            write_every=args.write_every,
            volume_m3=p.anode_gas_volume_m3,
            temperature_k=p.stack_temperature,
            gas_constant_j_mol_k=p.gas_constant,
            faraday_c_mol=p.faraday,
            membrane_area_m2=scaling.inferred_active_area_m2,
            fixed_charge_mol_m3=fixed_charge,
            target_total_pressure_pa=p.anode_target_total_pressure_pa,
            ambient_pressure_pa=p.pressure,
            purge_interval_as=p.tech.purge_interval_as,
            purge_clock_current_a=None,
            purge_duration_s=p.tech.lab_purge_duration_s,
            purge_reference_flow_slpm=p.tech.purge_rate_min_slpm_per_cell,
            current_scale_factor=scaling.area_scale_factor,
        )

        consumed = float(summary["cumulative_h2_consumed_mol"])
        purged = float(summary["cumulative_h2_purged_mol"])
        utilization = consumed / (consumed + purged) if consumed + purged > 0.0 else 0.0
        summary["hydrogen_utilization_consumed_over_consumed_plus_purged"] = utilization
        summary["manual_reference_hydrogen_utilization"] = (
            p.tech.h2_utilization_with_standard_purge
        )
        all_rows.extend(rows)
        summaries.append(summary)

        print(
            f"{regime}: RH={summary['final_relative_humidity']:.4f} "
            f"Iscale={scaling.area_scale_factor:.1f} "
            f"H2util={utilization:.4f} "
            f"Pmin={summary['min_total_pressure_pa'] / 1e5:.4f} bar",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-active-area-scaling",
        "closure_source": str(args.closure_csv),
        "calibration_regime": "nominal",
        "calibration_relative_humidity": args.calibration_rh,
        "patch_area_m2": scaling.patch_area_m2,
        "patch_reference_current_a": scaling.patch_reference_current_a,
        "cell_reference_current_a": scaling.cell_reference_current_a,
        "area_scale_factor": scaling.area_scale_factor,
        "inferred_active_area_m2": scaling.inferred_active_area_m2,
        "inferred_active_area_cm2": scaling.inferred_active_area_cm2,
        "anode_gas_volume_m3_per_cell": p.anode_gas_volume_m3,
        "anode_volume_ml_per_cm2": (
            p.anode_gas_volume_m3 * 1.0e6 / scaling.inferred_active_area_cm2
        ),
        "purge_reference_flow_slpm_per_cell": p.tech.purge_rate_min_slpm_per_cell,
        "purge_flow_slpm_per_cm2": (
            p.tech.purge_rate_min_slpm_per_cell / scaling.inferred_active_area_cm2
        ),
        "purge_clock_mode": "dynamic_cell_current",
        "manual_reference_hydrogen_utilization": (
            p.tech.h2_utilization_with_standard_purge
        ),
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "regimes": args.regimes,
        "summaries": summaries,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, all_rows)
    print(
        f"Inferred active area: {scaling.inferred_active_area_cm2:.2f} cm2 "
        f"(scale={scaling.area_scale_factor:.1f})"
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
