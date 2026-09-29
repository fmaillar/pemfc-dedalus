"""V0.7 lumped dynamic anode-water control volume.

This first V0.7 driver isolates anode gas-water dynamics from the expensive 3D
solve.  It uses the validated V0.6 RH_anode sensitivity table as a quasi-steady
closure for membrane water transfer:

    RH_anode -> membrane/anode water flux.

That flux is converted to mol/s and integrated in a finite anode gas volume.
The dynamic fixed point should approach the zero-flux RH inferred from the V0.6
static sensitivity campaign.  Hydrogen pressure/consumption and purge events
are intentionally deferred to the next V0.7 increments.
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
    lambda_flux_to_water_molar_rate,
    water_vapor_moles_from_relative_humidity,
)
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters

REGIMES = ("dry_high_load", "nominal", "wet_low_load")


def load_flux_closure(
    path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load RH_anode -> lambda-space flux closures from V0.6 results."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            regime = row["regime"]
            grouped.setdefault(regime, []).append(
                (
                    float(row["anode_relative_humidity"]),
                    float(row["final_anode_water_removal_flux_lambda_m_s"]),
                )
            )

    closure: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for regime, values in grouped.items():
        ordered = sorted(values)
        rh = np.asarray([item[0] for item in ordered], dtype=float)
        flux = np.asarray([item[1] for item in ordered], dtype=float)
        closure[regime] = (rh, flux)
    return closure


def interpolate_lambda_flux(
    relative_humidity: float,
    closure: tuple[np.ndarray, np.ndarray],
) -> float:
    """Linearly interpolate the validated V0.6 membrane/anode flux."""
    rh, flux = closure
    if rh.size < 2 or flux.size != rh.size:
        raise ValueError("flux closure must contain at least two matching points")
    return float(np.interp(relative_humidity, rh, flux))


def zero_flux_relative_humidity(
    closure: tuple[np.ndarray, np.ndarray],
) -> float | None:
    """Return the first linearly interpolated RH at which the closure flux is zero."""
    rh, flux = closure
    for index in range(1, rh.size):
        f0 = float(flux[index - 1])
        f1 = float(flux[index])
        if f0 == 0.0:
            return float(rh[index - 1])
        if f0 * f1 < 0.0:
            x0 = float(rh[index - 1])
            x1 = float(rh[index])
            return x0 - f0 * (x1 - x0) / (f1 - f0)
    if float(flux[-1]) == 0.0:
        return float(rh[-1])
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty V0.7 anode-volume table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_regime(
    regime: str,
    closure: tuple[np.ndarray, np.ndarray],
    *,
    initial_rh: float,
    stop_time_s: float,
    dt_s: float,
    volume_m3: float,
    temperature_k: float,
    gas_constant_j_mol_k: float,
    membrane_area_m2: float,
    fixed_charge_mol_m3: float,
    write_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Integrate one lumped anode-water trajectory."""
    initial_vapor = water_vapor_moles_from_relative_humidity(
        initial_rh,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    state = AnodeWaterState(
        vapor_mol=initial_vapor,
        liquid_mol=0.0,
        relative_humidity=initial_rh,
    )

    rows: list[dict[str, Any]] = []
    n_steps = int(np.ceil(stop_time_s / dt_s))

    for step in range(n_steps + 1):
        time_s = min(step * dt_s, stop_time_s)
        flux_lambda = interpolate_lambda_flux(state.relative_humidity, closure)
        water_source = lambda_flux_to_water_molar_rate(
            flux_lambda,
            membrane_area_m2=membrane_area_m2,
            fixed_charge_mol_m3=fixed_charge_mol_m3,
        )

        if step % write_every == 0 or step == n_steps:
            rows.append(
                {
                    "regime": regime,
                    "time_s": time_s,
                    "anode_relative_humidity": state.relative_humidity,
                    "water_flux_lambda_m_s": flux_lambda,
                    "water_source_mol_s": water_source,
                    "water_vapor_mol": state.vapor_mol,
                    "liquid_water_mol": state.liquid_mol,
                }
            )

        if step == n_steps:
            break

        actual_dt = min(dt_s, stop_time_s - time_s)
        state = advance_anode_water_state(
            state,
            water_source_mol_s=water_source,
            dt_s=actual_dt,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )

    equilibrium_rh = zero_flux_relative_humidity(closure)
    final_flux = interpolate_lambda_flux(state.relative_humidity, closure)
    summary = {
        "regime": regime,
        "initial_relative_humidity": initial_rh,
        "final_relative_humidity": state.relative_humidity,
        "final_water_flux_lambda_m_s": final_flux,
        "final_water_vapor_mol": state.vapor_mol,
        "final_liquid_water_mol": state.liquid_mol,
        "static_zero_flux_relative_humidity": equilibrium_rh,
        "absolute_rh_error_to_static_zero_flux": (
            None
            if equilibrium_rh is None
            else abs(state.relative_humidity - equilibrium_rh)
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-anode-volume.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-anode-volume.csv"),
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
    volume_m3 = p.anode_gas_volume_m3 if args.anode_gas_volume is None else args.anode_gas_volume
    if volume_m3 <= 0.0:
        parser.error("--anode-gas-volume must be positive")

    closure = load_flux_closure(args.closure_csv)
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
        rows, summary = simulate_regime(
            regime,
            closure[regime],
            initial_rh=args.initial_rh,
            stop_time_s=args.stop_time,
            dt_s=args.dt,
            volume_m3=volume_m3,
            temperature_k=p.stack_temperature,
            gas_constant_j_mol_k=p.gas_constant,
            membrane_area_m2=membrane_area,
            fixed_charge_mol_m3=fixed_charge,
            write_every=args.write_every,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    output = {
        "schema_version": 1,
        "model": "v07-anode-volume",
        "closure_source": str(args.closure_csv),
        "closure_kind": "piecewise-linear V0.6 RH_anode sensitivity",
        "anode_gas_volume_m3": volume_m3,
        "stack_temperature_k": p.stack_temperature,
        "initial_relative_humidity": args.initial_rh,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "membrane_area_m2": membrane_area,
        "fixed_charge_mol_m3": fixed_charge,
        "regimes": args.regimes,
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, all_rows)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    for summary in summaries:
        print(
            f"{summary['regime']}: RH {summary['initial_relative_humidity']:.4f} "
            f"-> {summary['final_relative_humidity']:.4f}, "
            f"static zero-flux RH={summary['static_zero_flux_relative_humidity']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
