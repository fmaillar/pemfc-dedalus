"""V0.7 purge-duration comparison with dynamic current-triggered purging.

This study combines the active-area scaling with a purge clock driven by the
instantaneous scaled cell current. It compares the laboratory 0.2 s opening
with the 0.5 s standard-purge limit from the Ballard reference.
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
        raise ValueError("cannot write empty purge-duration study table")
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
        "--durations",
        nargs="+",
        type=float,
        default=[0.2, 0.5],
    )
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-purge-duration-comparison.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-purge-duration-comparison.csv"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.initial_rh <= 1.0:
        parser.error("--initial-rh must be in [0, 1]")
    if args.stop_time <= 0.0 or args.dt <= 0.0:
        parser.error("--stop-time and --dt must be positive")
    if args.write_every < 1:
        parser.error("--write-every must be >= 1")
    if any(duration <= 0.0 for duration in args.durations):
        parser.error("--durations must all be positive")

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
    missing = [regime for regime in args.regimes if regime not in closure]
    if missing:
        parser.error(f"missing closure data for regimes: {', '.join(missing)}")
    if "nominal" not in closure:
        parser.error("nominal closure is required for active-area calibration")

    patch_area_m2 = p.length_x * p.length_y
    _, patch_reference_current_a = interpolate_flux_and_current(
        0.0,
        closure["nominal"],
    )
    scaling = infer_active_area_scaling(
        patch_area_m2=patch_area_m2,
        patch_reference_current_a=patch_reference_current_a,
        cell_reference_current_a=p.stack_current_a,
    )
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for duration_s in args.durations:
        for regime in args.regimes:
            trajectory, summary = simulate_transient_purge_regime(
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
                purge_duration_s=duration_s,
                purge_reference_flow_slpm=p.tech.purge_rate_min_slpm_per_cell,
                current_scale_factor=scaling.area_scale_factor,
            )

            for row in trajectory:
                rows.append(
                    {
                        "purge_duration_s": duration_s,
                        **row,
                    }
                )

            consumed = float(summary["cumulative_h2_consumed_mol"])
            purged = float(summary["cumulative_h2_purged_mol"])
            utilization = (
                consumed / (consumed + purged)
                if consumed + purged > 0.0
                else 0.0
            )
            events = summary["purge_events"]
            mean_purged_h2 = (
                sum(float(event["h2_purged_mol"]) for event in events)
                / len(events)
                if events
                else 0.0
            )
            mean_purged_water = (
                sum(float(event["water_purged_mol"]) for event in events)
                / len(events)
                if events
                else 0.0
            )
            result = {
                **summary,
                "purge_duration_s": duration_s,
                "hydrogen_utilization": utilization,
                "manual_reference_hydrogen_utilization": (
                    p.tech.h2_utilization_with_standard_purge
                ),
                "mean_h2_purged_per_event_mol": mean_purged_h2,
                "mean_water_purged_per_event_mol": mean_purged_water,
            }
            summaries.append(result)

            print(
                f"{regime} duration={duration_s:.1f}s: "
                f"purges={summary['purge_count']} "
                f"T={summary['mean_purge_period_s']} s "
                f"H2util={utilization:.4f}",
                flush=True,
            )

    output = {
        "schema_version": 1,
        "model": "v07-purge-duration-comparison",
        "closure_source": str(args.closure_csv),
        "purge_clock_mode": "dynamic_cell_current",
        "durations_s": args.durations,
        "patch_area_m2": scaling.patch_area_m2,
        "area_scale_factor": scaling.area_scale_factor,
        "inferred_active_area_m2": scaling.inferred_active_area_m2,
        "inferred_active_area_cm2": scaling.inferred_active_area_cm2,
        "purge_interval_as": p.tech.purge_interval_as,
        "purge_reference_flow_slpm": p.tech.purge_rate_min_slpm_per_cell,
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
    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
