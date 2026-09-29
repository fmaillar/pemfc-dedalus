"""V0.7 finite-duration pressure-driven purge for the dead-end anode.

This increment replaces the instantaneous purge/refill event with a valve-open
interval.  During a purge:
- gas leaves through a pressure-driven linear conductance;
- H2 consumption and membrane water transfer continue;
- the one-way H2 regulator remains active;
- H2 and H2O are removed according to the instantaneous gas composition.

The purge conductance is calibrated so that the Ballard minimum purge flow
(2.4 slpm per cell) is obtained at the nominal 0.36 barg pressure difference.
The 0.2 s laboratory purge duration is used as the default opening time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from pemfc_dedalus.anode import (
    AnodeWaterState,
    advance_anode_water_state,
    advance_pressure_regulated_hydrogen,
    anode_total_gas_pressure_pa,
    hydrogen_consumption_molar_rate,
    hydrogen_moles_for_total_pressure,
    lambda_flux_to_water_molar_rate,
    pressure_driven_purge_molar_rate,
    purge_pressure_conductance_mol_s_pa,
    remove_well_mixed_gas_moles,
    water_vapor_moles_from_relative_humidity,
)
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty transient purge trajectory")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_transient_purge_regime(
    regime: str,
    closure: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    initial_rh: float,
    stop_time_s: float,
    dt_s: float,
    write_every: int,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
    faraday_c_mol: float,
    membrane_area_m2: float,
    fixed_charge_mol_m3: float,
    target_total_pressure_pa: float,
    ambient_pressure_pa: float,
    purge_interval_as: float,
    purge_clock_current_a: float | None,
    purge_duration_s: float,
    purge_reference_flow_slpm: float,
    current_scale_factor: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Integrate H2/H2O inventories with a finite-duration purge valve."""
    if stop_time_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("stop_time_s and dt_s must be positive")
    if purge_interval_as <= 0.0:
        raise ValueError("purge_interval_as must be positive")
    if purge_clock_current_a is not None and purge_clock_current_a <= 0.0:
        raise ValueError("purge_clock_current_a must be positive when provided")
    if purge_duration_s <= 0.0:
        raise ValueError("purge_duration_s must be positive")
    if current_scale_factor <= 0.0:
        raise ValueError("current_scale_factor must be positive")
    if ambient_pressure_pa <= 0.0:
        raise ValueError("ambient_pressure_pa must be positive")

    initial_water_vapor = water_vapor_moles_from_relative_humidity(
        initial_rh,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    water_state = AnodeWaterState(
        vapor_mol=initial_water_vapor,
        liquid_mol=0.0,
        relative_humidity=initial_rh,
    )
    initial_hydrogen_mol = hydrogen_moles_for_total_pressure(
        target_total_pressure_pa,
        water_vapor_mol=initial_water_vapor,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    hydrogen_mol = initial_hydrogen_mol

    purge_conductance = purge_pressure_conductance_mol_s_pa(
        purge_reference_flow_slpm,
        reference_upstream_pressure_pa=target_total_pressure_pa,
        downstream_pressure_pa=ambient_pressure_pa,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )

    rows: list[dict[str, Any]] = []
    purge_events: list[dict[str, Any]] = []
    charge_since_purge_as = 0.0
    purge_remaining_s = 0.0
    active_event: dict[str, Any] | None = None

    cumulative_h2_inlet_mol = 0.0
    cumulative_h2_consumed_mol = 0.0
    cumulative_h2_purged_mol = 0.0
    cumulative_water_transfer_mol = 0.0
    cumulative_water_purged_mol = 0.0

    min_total_pressure_pa = target_total_pressure_pa
    max_total_pressure_pa = target_total_pressure_pa
    n_steps = int(np.ceil(stop_time_s / dt_s))

    for step in range(n_steps + 1):
        time_s = min(step * dt_s, stop_time_s)
        flux_lambda, patch_current_a = interpolate_flux_and_current(
            water_state.relative_humidity,
            closure,
        )
        cell_current_a = patch_current_a * current_scale_factor
        water_source = lambda_flux_to_water_molar_rate(
            flux_lambda,
            membrane_area_m2=membrane_area_m2,
            fixed_charge_mol_m3=fixed_charge_mol_m3,
        )
        p_total = anode_total_gas_pressure_pa(
            hydrogen_mol,
            water_state.vapor_mol,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        min_total_pressure_pa = min(min_total_pressure_pa, p_total)
        max_total_pressure_pa = max(max_total_pressure_pa, p_total)

        if step % write_every == 0 or step == n_steps:
            rows.append(
                {
                    "regime": regime,
                    "time_s": time_s,
                    "purge_open": purge_remaining_s > 0.0,
                    "purge_count": len(purge_events),
                    "anode_relative_humidity": water_state.relative_humidity,
                    "patch_current_a": patch_current_a,
                    "cell_current_a": cell_current_a,
                    "charge_since_purge_as": charge_since_purge_as,
                    "purge_clock_current_a": (
                        cell_current_a
                        if purge_clock_current_a is None
                        else purge_clock_current_a
                    ),
                    "hydrogen_mol": hydrogen_mol,
                    "water_vapor_mol": water_state.vapor_mol,
                    "liquid_water_mol": water_state.liquid_mol,
                    "total_pressure_pa": p_total,
                }
            )

        if step == n_steps:
            break

        actual_dt = min(dt_s, stop_time_s - time_s)

        water_state = advance_anode_water_state(
            water_state,
            water_source_mol_s=water_source,
            dt_s=actual_dt,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        cumulative_water_transfer_mol += water_source * actual_dt

        if purge_remaining_s > 0.0:
            consumption_rate = hydrogen_consumption_molar_rate(
                cell_current_a,
                faraday_c_mol,
            )
            consumed_h2 = min(
                consumption_rate * actual_dt,
                hydrogen_mol,
            )
            hydrogen_mol -= consumed_h2
            cumulative_h2_consumed_mol += consumed_h2

            pressure_before_outflow = anode_total_gas_pressure_pa(
                hydrogen_mol,
                water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            outflow_rate = pressure_driven_purge_molar_rate(
                pressure_before_outflow,
                ambient_pressure_pa,
                purge_conductance,
            )
            (
                hydrogen_mol,
                water_state,
                purged_hydrogen,
                purged_water,
            ) = remove_well_mixed_gas_moles(
                hydrogen_mol,
                water_state,
                gas_outflow_mol=outflow_rate * actual_dt,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            cumulative_h2_purged_mol += purged_hydrogen
            cumulative_water_purged_mol += purged_water

            pressure_after_outflow = anode_total_gas_pressure_pa(
                hydrogen_mol,
                water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            min_total_pressure_pa = min(
                min_total_pressure_pa,
                pressure_after_outflow,
            )

            target_hydrogen = hydrogen_moles_for_total_pressure(
                target_total_pressure_pa,
                water_vapor_mol=water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            refill_hydrogen = max(target_hydrogen - hydrogen_mol, 0.0)
            hydrogen_mol += refill_hydrogen
            cumulative_h2_inlet_mol += refill_hydrogen

            if active_event is None:
                raise RuntimeError("purge is active without an event record")
            active_event["h2_purged_mol"] += purged_hydrogen
            active_event["water_purged_mol"] += purged_water
            active_event["h2_refill_mol"] += refill_hydrogen
            active_event["min_pressure_pa"] = min(
                active_event["min_pressure_pa"],
                pressure_after_outflow,
            )
            active_event["max_outflow_mol_s"] = max(
                active_event["max_outflow_mol_s"],
                outflow_rate,
            )

            purge_remaining_s = max(purge_remaining_s - actual_dt, 0.0)
            if purge_remaining_s <= 1.0e-12:
                active_event["end_time_s"] = time_s + actual_dt
                active_event["rh_after"] = water_state.relative_humidity
                active_event["pressure_after_pa"] = anode_total_gas_pressure_pa(
                    hydrogen_mol,
                    water_state.vapor_mol,
                    volume_m3=volume_m3,
                    temperature_k=temperature_k,
                    gas_constant_j_mol_k=gas_constant_j_mol_k,
                )
                purge_events.append(active_event)
                active_event = None
        else:
            hydrogen_mol, inlet_rate, consumption_rate = (
                advance_pressure_regulated_hydrogen(
                    hydrogen_mol,
                    current_a=cell_current_a,
                    water_vapor_mol=water_state.vapor_mol,
                    dt_s=actual_dt,
                    target_total_pressure_pa=target_total_pressure_pa,
                    volume_m3=volume_m3,
                    temperature_k=temperature_k,
                    gas_constant_j_mol_k=gas_constant_j_mol_k,
                    faraday_c_mol=faraday_c_mol,
                )
            )
            cumulative_h2_inlet_mol += inlet_rate * actual_dt
            cumulative_h2_consumed_mol += consumption_rate * actual_dt

        clock_current_a = (
            cell_current_a
            if purge_clock_current_a is None
            else purge_clock_current_a
        )
        charge_since_purge_as += clock_current_a * actual_dt

        if (
            purge_remaining_s <= 0.0
            and charge_since_purge_as >= purge_interval_as
        ):
            charge_since_purge_as -= purge_interval_as
            purge_remaining_s = purge_duration_s
            active_event = {
                "event_index": len(purge_events) + 1,
                "start_time_s": time_s + actual_dt,
                "end_time_s": None,
                "rh_before": water_state.relative_humidity,
                "rh_after": None,
                "pressure_before_pa": anode_total_gas_pressure_pa(
                    hydrogen_mol,
                    water_state.vapor_mol,
                    volume_m3=volume_m3,
                    temperature_k=temperature_k,
                    gas_constant_j_mol_k=gas_constant_j_mol_k,
                ),
                "pressure_after_pa": None,
                "min_pressure_pa": float("inf"),
                "h2_purged_mol": 0.0,
                "water_purged_mol": 0.0,
                "h2_refill_mol": 0.0,
                "max_outflow_mol_s": 0.0,
            }

    final_total_water = water_state.vapor_mol + water_state.liquid_mol
    final_total_pressure = anode_total_gas_pressure_pa(
        hydrogen_mol,
        water_state.vapor_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    h2_balance_error = (
        initial_hydrogen_mol
        + cumulative_h2_inlet_mol
        - cumulative_h2_consumed_mol
        - cumulative_h2_purged_mol
        - hydrogen_mol
    )
    water_balance_error = (
        initial_water_vapor
        + cumulative_water_transfer_mol
        - cumulative_water_purged_mol
        - final_total_water
    )

    event_times = [float(event["start_time_s"]) for event in purge_events]
    periods = [
        event_times[index] - event_times[index - 1]
        for index in range(1, len(event_times))
    ]

    summary = {
        "regime": regime,
        "purge_count": len(purge_events),
        "purge_interval_as": purge_interval_as,
        "purge_clock_mode": (
            "dynamic_cell_current"
            if purge_clock_current_a is None
            else "fixed_current"
        ),
        "purge_clock_current_a": purge_clock_current_a,
        "expected_purge_period_s": (
            None
            if purge_clock_current_a is None
            else purge_interval_as / purge_clock_current_a
        ),
        "purge_duration_s": purge_duration_s,
        "purge_reference_flow_slpm": purge_reference_flow_slpm,
        "purge_conductance_mol_s_pa": purge_conductance,
        "first_purge_time_s": event_times[0] if event_times else None,
        "mean_purge_period_s": mean(periods) if periods else None,
        "final_relative_humidity": water_state.relative_humidity,
        "final_total_pressure_pa": final_total_pressure,
        "min_total_pressure_pa": min_total_pressure_pa,
        "max_total_pressure_pa": max_total_pressure_pa,
        "cumulative_h2_inlet_mol": cumulative_h2_inlet_mol,
        "cumulative_h2_consumed_mol": cumulative_h2_consumed_mol,
        "cumulative_h2_purged_mol": cumulative_h2_purged_mol,
        "cumulative_water_transfer_mol": cumulative_water_transfer_mol,
        "cumulative_water_purged_mol": cumulative_water_purged_mol,
        "h2_balance_error_mol": h2_balance_error,
        "water_balance_error_mol": water_balance_error,
        "purge_events": purge_events,
    }
    return rows, summary


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
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument("--anode-gas-volume", type=float, default=None)
    parser.add_argument("--target-total-pressure", type=float, default=None)
    parser.add_argument("--ambient-pressure", type=float, default=None)
    parser.add_argument("--purge-interval-as", type=float, default=None)
    parser.add_argument("--purge-clock-current", type=float, default=None)
    parser.add_argument("--purge-duration", type=float, default=None)
    parser.add_argument("--purge-reference-flow-slpm", type=float, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-anode-transient-purge.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-anode-transient-purge.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    volume_m3 = p.anode_gas_volume_m3 if args.anode_gas_volume is None else args.anode_gas_volume
    target_pressure = (
        p.anode_target_total_pressure_pa
        if args.target_total_pressure is None
        else args.target_total_pressure
    )
    ambient_pressure = (
        p.pressure
        if args.ambient_pressure is None
        else args.ambient_pressure
    )
    purge_interval_as = (
        p.tech.purge_interval_as
        if args.purge_interval_as is None
        else args.purge_interval_as
    )
    purge_clock_current = (
        p.stack_current_a
        if args.purge_clock_current is None
        else args.purge_clock_current
    )
    purge_duration = (
        p.tech.lab_purge_duration_s
        if args.purge_duration is None
        else args.purge_duration
    )
    purge_reference_flow = (
        p.tech.purge_rate_min_slpm_per_cell
        if args.purge_reference_flow_slpm is None
        else args.purge_reference_flow_slpm
    )

    if not 0.0 <= args.initial_rh <= 1.0:
        parser.error("--initial-rh must be in [0, 1]")
    if volume_m3 <= 0.0:
        parser.error("--anode-gas-volume must be positive")
    if target_pressure <= ambient_pressure:
        parser.error("target pressure must exceed ambient pressure")
    if args.dt <= 0.0 or args.stop_time <= 0.0:
        parser.error("--dt and --stop-time must be positive")
    if args.write_every < 1:
        parser.error("--write-every must be >= 1")

    closure = load_h2_closure(args.closure_csv)
    missing = [regime for regime in args.regimes if regime not in closure]
    if missing:
        parser.error(f"missing closure data for regimes: {', '.join(missing)}")

    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    membrane_area = p.length_x * p.length_y

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
            volume_m3=volume_m3,
            temperature_k=p.stack_temperature,
            gas_constant_j_mol_k=p.gas_constant,
            faraday_c_mol=p.faraday,
            membrane_area_m2=membrane_area,
            fixed_charge_mol_m3=fixed_charge,
            target_total_pressure_pa=target_pressure,
            ambient_pressure_pa=ambient_pressure,
            purge_interval_as=purge_interval_as,
            purge_clock_current_a=purge_clock_current,
            purge_duration_s=purge_duration,
            purge_reference_flow_slpm=purge_reference_flow,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{regime}: purges={summary['purge_count']} "
            f"T={summary['mean_purge_period_s']} s "
            f"Pmin={summary['min_total_pressure_pa'] / 1e5:.4f} bar "
            f"Pmax={summary['max_total_pressure_pa'] / 1e5:.4f} bar",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-anode-transient-purge",
        "closure_source": str(args.closure_csv),
        "anode_gas_volume_m3": volume_m3,
        "target_total_pressure_pa": target_pressure,
        "ambient_pressure_pa": ambient_pressure,
        "purge_interval_as": purge_interval_as,
        "purge_clock_current_a": purge_clock_current,
        "purge_duration_s": purge_duration,
        "purge_reference_flow_slpm": purge_reference_flow,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "regimes": args.regimes,
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, all_rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
