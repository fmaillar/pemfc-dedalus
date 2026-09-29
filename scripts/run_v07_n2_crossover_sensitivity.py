"""V0.7 nitrogen crossover sensitivity study.

This campaign varies the imposed cathode-to-anode N2 crossover flux while
keeping the validated full-cell scaling, dynamic 2300 A.s purge clock and
standard 0.5 s / 2.4 slpm purge configuration unchanged.

No H2-dilution feedback is applied to electrochemistry in this increment.
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

DEFAULT_FLUXES = [0.0, 2.5e-7, 5.0e-7, 1.0e-6, 2.0e-6, 5.0e-6]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty N2 sensitivity table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def monotonic_non_decreasing(values: list[float], tol: float = 1.0e-12) -> bool:
    return all(
        values[index + 1] + tol >= values[index]
        for index in range(len(values) - 1)
    )


def monotonic_non_increasing(values: list[float], tol: float = 1.0e-12) -> bool:
    return all(
        values[index + 1] <= values[index] + tol
        for index in range(len(values) - 1)
    )


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
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-n2-crossover-sensitivity.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-n2-crossover-sensitivity.csv"),
    )
    args = parser.parse_args()

    if not args.fluxes:
        parser.error("--fluxes must not be empty")
    if any(flux < 0.0 for flux in args.fluxes):
        parser.error("--fluxes must all be non-negative")
    if sorted(args.fluxes) != args.fluxes:
        parser.error("--fluxes must be sorted in ascending order")
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
    summaries: list[dict[str, Any]] = []

    for flux in args.fluxes:
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
            )
            row = {
                "regime": regime,
                "n2_crossover_flux_mol_m2_s": flux,
                "n2_crossover_rate_mol_s": summary["n2_crossover_rate_mol_s"],
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
                "final_nitrogen_mol": summary["final_nitrogen_mol"],
                "cumulative_n2_crossover_mol": (
                    summary["cumulative_n2_crossover_mol"]
                ),
                "cumulative_n2_purged_mol": (
                    summary["cumulative_n2_purged_mol"]
                ),
                "min_total_pressure_pa": summary["min_total_pressure_pa"],
                "max_total_pressure_pa": summary["max_total_pressure_pa"],
                "h2_balance_error_mol": summary["h2_balance_error_mol"],
                "n2_balance_error_mol": summary["n2_balance_error_mol"],
                "water_balance_error_mol": summary["water_balance_error_mol"],
            }
            rows.append(row)
            summaries.append(row)
            print(
                f"{regime} J_N2={flux:.2e}: "
                f"xN2,max={row['max_nitrogen_mole_fraction']:.5f} "
                f"xH2,min={row['min_hydrogen_mole_fraction']:.5f}",
                flush=True,
            )

    monotonicity: dict[str, dict[str, bool]] = {}
    for regime in args.regimes:
        regime_rows = [row for row in rows if row["regime"] == regime]
        x_n2 = [
            float(row["max_nitrogen_mole_fraction"])
            for row in regime_rows
        ]
        x_h2 = [
            float(row["min_hydrogen_mole_fraction"])
            for row in regime_rows
        ]
        monotonicity[regime] = {
            "max_n2_non_decreasing": monotonic_non_decreasing(x_n2),
            "min_h2_non_increasing": monotonic_non_increasing(x_h2),
        }

    output = {
        "schema_version": 1,
        "model": "v07-n2-crossover-sensitivity",
        "closure_source": str(args.closure_csv),
        "area_scale_factor": scaling.area_scale_factor,
        "inferred_active_area_cm2": scaling.inferred_active_area_cm2,
        "fluxes_mol_m2_s": args.fluxes,
        "purge_duration_s": p.tech.purge_duration_max_s,
        "purge_reference_flow_slpm": p.tech.purge_rate_min_slpm_per_cell,
        "purge_interval_as": p.tech.purge_interval_as,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "regimes": args.regimes,
        "monotonicity": monotonicity,
        "summaries": summaries,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
