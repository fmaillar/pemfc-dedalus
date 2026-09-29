"""V0.7 hydrogen-dilution electrochemical feedback sensitivity.

This campaign keeps the reduced N2 crossover model and standard purge
configuration, then multiplies the V0.6 RH-based current by

    f_H2 = (p_H2 / p_H2,ref) ** gamma

where p_H2,ref is evaluated at the same water state with N2 removed.  This
normalization isolates inert-gas dilution and avoids double-counting the RH
dependence already embedded in the V0.6 closure.

The exponent gamma is a screening parameter, not a calibrated HOR kinetic
coefficient.
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
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime

DEFAULT_FLUXES = [0.0, 1.0e-6, 2.0e-6, 5.0e-6]
DEFAULT_EXPONENTS = [0.0, 0.5, 1.0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty H2 feedback table")
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
    parser.add_argument(
        "--fluxes",
        nargs="+",
        type=float,
        default=DEFAULT_FLUXES,
    )
    parser.add_argument(
        "--exponents",
        nargs="+",
        type=float,
        default=DEFAULT_EXPONENTS,
    )
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-h2-feedback-sensitivity.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-h2-feedback-sensitivity.csv"),
    )
    args = parser.parse_args()

    if not args.fluxes or any(flux < 0.0 for flux in args.fluxes):
        parser.error("--fluxes must be non-empty and non-negative")
    if not args.exponents or any(value < 0.0 for value in args.exponents):
        parser.error("--exponents must be non-empty and non-negative")
    if args.stop_time <= 0.0 or args.dt <= 0.0:
        parser.error("--stop-time and --dt must be positive")
    if args.write_every < 1:
        parser.error("--write-every must be >= 1")
    if not 0.0 <= args.initial_rh <= 1.0:
        parser.error("--initial-rh must be in [0, 1]")

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
    missing = [regime for regime in args.regimes if regime not in closure]
    if missing:
        parser.error(f"missing closure data for regimes: {', '.join(missing)}")

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

    rows: list[dict[str, Any]] = []

    for flux in args.fluxes:
        for exponent in args.exponents:
            for regime in args.regimes:
                _, summary = simulate_nitrogen_regime(
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
                    active_area_m2=scaling.inferred_active_area_m2,
                    fixed_charge_mol_m3=fixed_charge,
                    target_total_pressure_pa=p.anode_target_total_pressure_pa,
                    ambient_pressure_pa=p.pressure,
                    purge_interval_as=p.tech.purge_interval_as,
                    purge_duration_s=p.tech.purge_duration_max_s,
                    purge_reference_flow_slpm=p.tech.purge_rate_min_slpm_per_cell,
                    current_scale_factor=scaling.area_scale_factor,
                    n2_crossover_flux_mol_m2_s=flux,
                    hydrogen_feedback_exponent=exponent,
                )
                row = {
                    "regime": regime,
                    "n2_crossover_flux_mol_m2_s": flux,
                    "hydrogen_feedback_exponent": exponent,
                    "min_hydrogen_feedback_factor": (
                        summary["min_hydrogen_feedback_factor"]
                    ),
                    "mean_cell_current_a": summary["mean_cell_current_a"],
                    "min_cell_current_a": summary["min_cell_current_a"],
                    "purge_count": summary["purge_count"],
                    "first_purge_time_s": summary["first_purge_time_s"],
                    "mean_purge_period_s": summary["mean_purge_period_s"],
                    "max_nitrogen_mole_fraction": (
                        summary["max_nitrogen_mole_fraction"]
                    ),
                    "min_hydrogen_mole_fraction": (
                        summary["min_hydrogen_mole_fraction"]
                    ),
                    "final_nitrogen_mole_fraction": (
                        summary["final_nitrogen_mole_fraction"]
                    ),
                    "cumulative_h2_consumed_mol": (
                        summary["cumulative_h2_consumed_mol"]
                    ),
                    "cumulative_h2_purged_mol": (
                        summary["cumulative_h2_purged_mol"]
                    ),
                    "cumulative_n2_purged_mol": (
                        summary["cumulative_n2_purged_mol"]
                    ),
                    "h2_balance_error_mol": summary["h2_balance_error_mol"],
                    "n2_balance_error_mol": summary["n2_balance_error_mol"],
                    "water_balance_error_mol": summary["water_balance_error_mol"],
                }
                rows.append(row)
                print(
                    f"{regime} J_N2={flux:.2e} gamma={exponent:.2f}: "
                    f"Imean={row['mean_cell_current_a']:.3f} A "
                    f"fmin={row['min_hydrogen_feedback_factor']:.5f} "
                    f"purges={row['purge_count']}",
                    flush=True,
                )

    output = {
        "schema_version": 1,
        "model": "v07-h2-feedback-sensitivity",
        "closure_source": str(args.closure_csv),
        "area_scale_factor": scaling.area_scale_factor,
        "inferred_active_area_cm2": scaling.inferred_active_area_cm2,
        "fluxes_mol_m2_s": args.fluxes,
        "feedback_exponents": args.exponents,
        "feedback_law": "min(p_H2 / p_H2_ref_same_RH_no_N2, 1) ** gamma",
        "purge_duration_s": p.tech.purge_duration_max_s,
        "purge_reference_flow_slpm": p.tech.purge_rate_min_slpm_per_cell,
        "purge_interval_as": p.tech.purge_interval_as,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "regimes": args.regimes,
        "summaries": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
