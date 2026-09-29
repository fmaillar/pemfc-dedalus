"""V0.7 discrete purge model for the lumped dead-end anode.

This increment adds periodic purge events to the validated V0.7 H2/H2O control
volume.  Purges are triggered by the Ballard charge-throughput reference
(2300 A.s) using the stack current, not the representative patch current.

Each purge is represented as a perfectly mixed gas-volume exchange.  The
fraction of gas removed is:

    f = 1 - exp(-V_purge / V_anode)

The purge removes the same fraction of H2 and gas-phase H2O.  The one-way H2
regulator then refills only enough H2 to recover the target total pressure.
Nitrogen and pressure-dependent electrochemistry remain outside this increment.
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
    hydrogen_moles_for_total_pressure,
    ideal_gas_partial_pressure_pa,
    lambda_flux_to_water_molar_rate,
    mixed_gas_purge_fraction,
    purge_anode_gas,
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
        raise ValueError("cannot write empty V0.7 purge trajectory table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_purge_regime(
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
    purge_interval_as: float,
    purge_clock_current_a: float,
    purge_exchange_volume_m3: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Integrate H2/H2O inventories with discrete charge-triggered purges."""
    if purge_interval_as <= 0.0:
        raise ValueError("purge_interval_as must be positive")
    if purge_clock_current_a <= 0.0:
        raise ValueError("purge_clock_current_a must be positive")

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

    purge_fraction = mixed_gas_purge_fraction(
        purge_exchange_volume_m3,
        volume_m3,
    )
    rows: list[dict[str, Any]] = []
    purge_events: list[dict[str, Any]] = []

    cumulative_h2_inlet_mol = 0.0
    cumulative_h2_consumed_mol = 0.0
    cumulative_h2_purged_mol = 0.0
    cumulative_water_transfer_mol = 0.0
    cumulative_water_purged_mol = 0.0
    charge_since_purge_as = 0.0
    max_total_pressure_pa = target_total_pressure_pa
    min_relative_humidity = initial_rh
    max_relative_humidity = initial_rh

    n_steps = int(np.ceil(stop_time_s / dt_s))

    for step in range(n_steps + 1):
        time_s = min(step * dt_s, stop_time_s)
        flux_lambda, current_a = interpolate_flux_and_current(
            water_state.relative_humidity,
            closure,
        )
        water_source = lambda_flux_to_water_molar_rate(
            flux_lambda,
            membrane_area_m2=membrane_area_m2,
            fixed_charge_mol_m3=fixed_charge_mol_m3,
        )

        p_h2 = ideal_gas_partial_pressure_pa(
            hydrogen_mol,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        p_h2o = ideal_gas_partial_pressure_pa(
            water_state.vapor_mol,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        p_total = anode_total_gas_pressure_pa(
            hydrogen_mol,
            water_state.vapor_mol,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        max_total_pressure_pa = max(max_total_pressure_pa, p_total)
        min_relative_humidity = min(
            min_relative_humidity,
            water_state.relative_humidity,
        )
        max_relative_humidity = max(
            max_relative_humidity,
            water_state.relative_humidity,
        )

        if step % write_every == 0 or step == n_steps:
            rows.append(
                {
                    "regime": regime,
                    "time_s": time_s,
                    "anode_relative_humidity": water_state.relative_humidity,
                    "patch_current_a": current_a,
                    "purge_clock_current_a": purge_clock_current_a,
                    "charge_since_purge_as": charge_since_purge_as,
                    "purge_count": len(purge_events),
                    "water_flux_lambda_m_s": flux_lambda,
                    "hydrogen_mol": hydrogen_mol,
                    "water_vapor_mol": water_state.vapor_mol,
                    "liquid_water_mol": water_state.liquid_mol,
                    "hydrogen_partial_pressure_pa": p_h2,
                    "water_vapor_partial_pressure_pa": p_h2o,
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

        hydrogen_mol, inlet_rate, consumption_rate = (
            advance_pressure_regulated_hydrogen(
                hydrogen_mol,
                current_a=current_a,
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

        charge_since_purge_as += purge_clock_current_a * actual_dt

        while charge_since_purge_as >= purge_interval_as:
            event_time_s = time_s + actual_dt
            pressure_before = anode_total_gas_pressure_pa(
                hydrogen_mol,
                water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            max_total_pressure_pa = max(
                max_total_pressure_pa,
                pressure_before,
            )
            rh_before = water_state.relative_humidity

            (
                hydrogen_mol,
                water_state,
                purged_hydrogen,
                purged_water,
            ) = purge_anode_gas(
                hydrogen_mol,
                water_state,
                purge_fraction=purge_fraction,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            cumulative_h2_purged_mol += purged_hydrogen
            cumulative_water_purged_mol += purged_water

            pressure_after_purge = anode_total_gas_pressure_pa(
                hydrogen_mol,
                water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
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

            pressure_after_refill = anode_total_gas_pressure_pa(
                hydrogen_mol,
                water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )

            purge_events.append(
                {
                    "event_index": len(purge_events) + 1,
                    "time_s": event_time_s,
                    "rh_before": rh_before,
                    "rh_after": water_state.relative_humidity,
                    "pressure_before_pa": pressure_before,
                    "pressure_after_purge_pa": pressure_after_purge,
                    "pressure_after_refill_pa": pressure_after_refill,
                    "purged_hydrogen_mol": purged_hydrogen,
                    "purged_water_mol": purged_water,
                    "refill_hydrogen_mol": refill_hydrogen,
                }
            )
            charge_since_purge_as -= purge_interval_as

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

    event_times = [float(event["time_s"]) for event in purge_events]
    periods = [
        event_times[index] - event_times[index - 1]
        for index in range(1, len(event_times))
    ]

    summary = {
        "regime": regime,
        "purge_count": len(purge_events),
        "purge_fraction": purge_fraction,
        "purge_interval_as": purge_interval_as,
        "purge_clock_current_a": purge_clock_current_a,
        "expected_purge_period_s": purge_interval_as / purge_clock_current_a,
        "first_purge_time_s": event_times[0] if event_times else None,
        "mean_purge_period_s": mean(periods) if periods else None,
        "final_relative_humidity": water_state.relative_humidity,
        "min_relative_humidity": min_relative_humidity,
        "max_relative_humidity": max_relative_humidity,
        "final_total_pressure_pa": final_total_pressure,
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
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--write-every", type=int, default=10)
    parser.add_argument("--anode-gas-volume", type=float, default=None)
    parser.add_argument("--target-total-pressure", type=float, default=None)
    parser.add_argument("--purge-interval-as", type=float, default=None)
    parser.add_argument("--purge-clock-current", type=float, default=None)
    parser.add_argument("--purge-exchange-volume", type=float, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-anode-purge.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-anode-purge.csv"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.initial_rh <= 1.0:
        parser.error("--initial-rh must be in [0, 1]")
    if args.stop_time <= 0.0:
        parser.error("--stop-time must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.write_every < 1:
        parser.error("--write-every must be >= 1")

    p = CathodeParameters()
    volume_m3 = (
        p.anode_gas_volume_m3
        if args.anode_gas_volume is None
        else args.anode_gas_volume
    )
    target_pressure = (
        p.anode_target_total_pressure_pa
        if args.target_total_pressure is None
        else args.target_total_pressure
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
    purge_exchange_volume = (
        p.tech.purge_volume_per_cell_m3
        if args.purge_exchange_volume is None
        else args.purge_exchange_volume
    )

    if volume_m3 <= 0.0:
        parser.error("--anode-gas-volume must be positive")
    if target_pressure <= 0.0:
        parser.error("--target-total-pressure must be positive")
    if purge_interval_as <= 0.0:
        parser.error("--purge-interval-as must be positive")
    if purge_clock_current <= 0.0:
        parser.error("--purge-clock-current must be positive")
    if purge_exchange_volume < 0.0:
        parser.error("--purge-exchange-volume must be non-negative")

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
        rows, summary = simulate_purge_regime(
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
            purge_interval_as=purge_interval_as,
            purge_clock_current_a=purge_clock_current,
            purge_exchange_volume_m3=purge_exchange_volume,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{regime}: purges={summary['purge_count']} "
            f"Tpurge={summary['mean_purge_period_s']} s "
            f"RH={summary['final_relative_humidity']:.4f} "
            f"Pmax={summary['max_total_pressure_pa'] / 1e5:.4f} bar",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-anode-purge",
        "closure_source": str(args.closure_csv),
        "anode_gas_volume_m3": volume_m3,
        "target_total_pressure_pa": target_pressure,
        "purge_interval_as": purge_interval_as,
        "purge_clock_current_a": purge_clock_current,
        "purge_exchange_volume_m3": purge_exchange_volume,
        "purge_fraction": mixed_gas_purge_fraction(
            purge_exchange_volume,
            volume_m3,
        ),
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
