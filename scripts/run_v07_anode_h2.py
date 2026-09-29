"""V0.7 dynamic dead-end anode H2 inventory without purge.

This driver extends the validated V0.7 water-volume model with:
- electrochemical H2 consumption from Faraday's law;
- a finite H2 inventory in the anode gas volume;
- ideal-gas H2 and H2O partial pressures;
- an idealized one-way pressure regulator at the Ballard optimal supply pressure.

The regulator may add H2 but cannot remove gas from the dead-end volume.  Purge,
nitrogen crossover and pressure-dependent electrochemistry remain intentionally
absent in this increment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
    water_vapor_moles_from_relative_humidity,
)
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters

REGIMES = ("dry_high_load", "nominal", "wet_low_load")


def load_h2_closure(
    path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load RH_anode -> (water flux, current) closures from V0.6 results."""
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["regime"], []).append(
                (
                    float(row["anode_relative_humidity"]),
                    float(row["final_anode_water_removal_flux_lambda_m_s"]),
                    float(row["final_current_a"]),
                )
            )

    closure: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for regime, values in grouped.items():
        ordered = sorted(values)
        closure[regime] = (
            np.asarray([item[0] for item in ordered], dtype=float),
            np.asarray([item[1] for item in ordered], dtype=float),
            np.asarray([item[2] for item in ordered], dtype=float),
        )
    return closure


def interpolate_flux_and_current(
    relative_humidity: float,
    closure: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """Linearly interpolate V0.6 water flux and representative-cell current."""
    rh, flux, current = closure
    if rh.size < 2 or flux.size != rh.size or current.size != rh.size:
        raise ValueError("closure arrays must contain at least two matching points")
    return (
        float(np.interp(relative_humidity, rh, flux)),
        float(np.interp(relative_humidity, rh, current)),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty V0.7 H2 trajectory table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_h2_regime(
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Integrate coupled lumped H2 and water inventories for one regime."""
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
    hydrogen_mol = hydrogen_moles_for_total_pressure(
        target_total_pressure_pa,
        water_vapor_mol=initial_water_vapor,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )

    rows: list[dict[str, Any]] = []
    cumulative_h2_inlet_mol = 0.0
    cumulative_h2_consumed_mol = 0.0
    max_total_pressure_pa = target_total_pressure_pa
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

        if step % write_every == 0 or step == n_steps:
            rows.append(
                {
                    "regime": regime,
                    "time_s": time_s,
                    "anode_relative_humidity": water_state.relative_humidity,
                    "current_a": current_a,
                    "water_flux_lambda_m_s": flux_lambda,
                    "water_source_mol_s": water_source,
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

    final_p_h2 = ideal_gas_partial_pressure_pa(
        hydrogen_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    final_p_total = anode_total_gas_pressure_pa(
        hydrogen_mol,
        water_state.vapor_mol,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    summary = {
        "regime": regime,
        "initial_relative_humidity": initial_rh,
        "final_relative_humidity": water_state.relative_humidity,
        "final_hydrogen_mol": hydrogen_mol,
        "final_water_vapor_mol": water_state.vapor_mol,
        "final_liquid_water_mol": water_state.liquid_mol,
        "final_hydrogen_partial_pressure_pa": final_p_h2,
        "final_total_pressure_pa": final_p_total,
        "max_total_pressure_pa": max_total_pressure_pa,
        "target_total_pressure_pa": target_total_pressure_pa,
        "cumulative_h2_inlet_mol": cumulative_h2_inlet_mol,
        "cumulative_h2_consumed_mol": cumulative_h2_consumed_mol,
        "net_h2_inventory_change_mol": (
            hydrogen_mol
            - hydrogen_moles_for_total_pressure(
                target_total_pressure_pa,
                water_vapor_mol=initial_water_vapor,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
        ),
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
    parser.add_argument("--stop-time", type=float, default=15000.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--write-every", type=int, default=10)
    parser.add_argument("--anode-gas-volume", type=float, default=None)
    parser.add_argument("--target-total-pressure", type=float, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-anode-h2.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-anode-h2.csv"),
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
    if volume_m3 <= 0.0:
        parser.error("--anode-gas-volume must be positive")
    if target_pressure <= 0.0:
        parser.error("--target-total-pressure must be positive")

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
        rows, summary = simulate_h2_regime(
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
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{regime}: RH={summary['final_relative_humidity']:.4f} "
            f"P={summary['final_total_pressure_pa'] / 1e5:.4f} bar "
            f"Pmax={summary['max_total_pressure_pa'] / 1e5:.4f} bar",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-anode-h2",
        "closure_source": str(args.closure_csv),
        "anode_gas_volume_m3": volume_m3,
        "stack_temperature_k": p.stack_temperature,
        "initial_relative_humidity": args.initial_rh,
        "target_total_pressure_pa": target_pressure,
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
