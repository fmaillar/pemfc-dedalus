"""V0.7 reduced N2 crossover and purge model.

This increment adds an explicit nitrogen inventory to the full-cell-scaled
anode model.  Nitrogen crosses from cathode to anode through a prescribed
molar flux, accumulates between purges, contributes to total pressure, dilutes
H2, and is removed with the well-mixed H2/N2/H2O gas during each standard
0.5 s purge.

Electrochemistry still depends only on RH through the validated V0.6 closure;
there is deliberately no feedback from p_H2 or x_H2 yet.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from pemfc_dedalus.anode import (
    AnodeWaterState,
    advance_anode_water_state,
    hydrogen_consumption_molar_rate,
    lambda_flux_to_water_molar_rate,
    pressure_driven_purge_molar_rate,
    purge_pressure_conductance_mol_s_pa,
    water_vapor_moles_from_relative_humidity,
)
from pemfc_dedalus.anode_nitrogen import (
    AnodeGasState,
    hydrogen_moles_for_pressure_with_nitrogen,
    nitrogen_crossover_molar_rate,
    nitrogen_pressure_driven_flux,
    remove_well_mixed_h2_n2_h2o,
    total_gas_pressure_with_nitrogen_pa,
)
from pemfc_dedalus.hydrogen_feedback import (
    hydrogen_partial_pressure_feedback_factor,
)
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty nitrogen trajectory")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_nitrogen_regime(
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
    active_area_m2: float,
    fixed_charge_mol_m3: float,
    target_total_pressure_pa: float,
    ambient_pressure_pa: float,
    purge_interval_as: float,
    purge_duration_s: float,
    purge_reference_flow_slpm: float,
    current_scale_factor: float,
    n2_crossover_flux_mol_m2_s: float,
    hydrogen_feedback_exponent: float = 0.0,
    n2_crossover_permeance_mol_m2_s_pa: float | None = None,
    cathode_n2_partial_pressure_pa: float | None = None,
    n2_permeance_model: Callable[[float], float] | None = None,
    n2_state_permeance_model: Callable[[float, float], float] | None = None,
    n2_state_temperature_permeance_model: (
        Callable[[float, float, float], float] | None
    ) = None,
    cathode_n2_partial_pressure_model: Callable[[float], float] | None = None,
    temperature_model_k: Callable[[float], float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Integrate H2/N2/H2O inventories with charge-triggered standard purges."""
    if stop_time_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("stop_time_s and dt_s must be positive")
    if purge_interval_as <= 0.0 or purge_duration_s <= 0.0:
        raise ValueError("purge parameters must be positive")
    if current_scale_factor <= 0.0:
        raise ValueError("current_scale_factor must be positive")
    if hydrogen_feedback_exponent < 0.0:
        raise ValueError("hydrogen_feedback_exponent must be non-negative")
    if n2_crossover_permeance_mol_m2_s_pa is not None:
        if n2_crossover_permeance_mol_m2_s_pa < 0.0:
            raise ValueError("n2 crossover permeance must be non-negative")
    permeance_models = sum(
        model is not None
        for model in (
            n2_permeance_model,
            n2_state_permeance_model,
            n2_state_temperature_permeance_model,
        )
    )
    if permeance_models > 1:
        raise ValueError("provide only one dynamic N2 permeance model")
    pressure_driven = (
        n2_crossover_permeance_mol_m2_s_pa is not None
        or n2_permeance_model is not None
        or n2_state_permeance_model is not None
        or n2_state_temperature_permeance_model is not None
    )
    if pressure_driven:
        if (
            cathode_n2_partial_pressure_pa is None
            and cathode_n2_partial_pressure_model is None
        ):
            raise ValueError("cathode N2 partial pressure is required")
        if (
            cathode_n2_partial_pressure_pa is not None
            and cathode_n2_partial_pressure_pa <= 0.0
        ):
            raise ValueError("cathode N2 partial pressure must be positive")

    initial_temperature_k = (
        temperature_k
        if temperature_model_k is None
        else float(temperature_model_k(0.0))
    )
    if initial_temperature_k <= 0.0:
        raise ValueError("temperature model must return positive values")

    initial_water = water_vapor_moles_from_relative_humidity(
        initial_rh,
        volume_m3=volume_m3,
        temperature_k=initial_temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    water_state = AnodeWaterState(
        vapor_mol=initial_water,
        liquid_mol=0.0,
        relative_humidity=initial_rh,
    )
    hydrogen_mol = hydrogen_moles_for_pressure_with_nitrogen(
        target_total_pressure_pa,
        nitrogen_mol=0.0,
        water_vapor_mol=initial_water,
        volume_m3=volume_m3,
        temperature_k=initial_temperature_k,
        gas_constant_j_mol_k=gas_constant_j_mol_k,
    )
    initial_hydrogen_mol = hydrogen_mol
    nitrogen_mol = 0.0

    constant_n2_source_rate = nitrogen_crossover_molar_rate(
        n2_crossover_flux_mol_m2_s,
        active_area_m2,
    )
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
    cumulative_n2_crossover_mol = 0.0
    cumulative_n2_purged_mol = 0.0
    cumulative_water_transfer_mol = 0.0
    cumulative_water_purged_mol = 0.0

    min_total_pressure_pa = target_total_pressure_pa
    max_total_pressure_pa = target_total_pressure_pa
    max_n2_mole_fraction = 0.0
    min_h2_mole_fraction = 1.0
    min_hydrogen_feedback_factor = 1.0
    min_cell_current_a = float("inf")
    min_n2_crossover_flux = float("inf")
    max_n2_crossover_flux = 0.0
    min_n2_permeance = float("inf")
    max_n2_permeance = 0.0

    n_steps = int(np.ceil(stop_time_s / dt_s))

    for step in range(n_steps + 1):
        time_s = min(step * dt_s, stop_time_s)
        step_temperature_k = (
            temperature_k
            if temperature_model_k is None
            else float(temperature_model_k(time_s))
        )
        if step_temperature_k <= 0.0:
            raise ValueError("temperature model must return positive values")
        flux_lambda, patch_current_a = interpolate_flux_and_current(
            water_state.relative_humidity,
            closure,
        )
        base_cell_current_a = patch_current_a * current_scale_factor

        gas_total = hydrogen_mol + nitrogen_mol + water_state.vapor_mol
        h2_mole_fraction = hydrogen_mol / gas_total if gas_total > 0.0 else 0.0
        n2_mole_fraction = nitrogen_mol / gas_total if gas_total > 0.0 else 0.0
        hydrogen_partial_pressure_pa = (
            hydrogen_mol
            * gas_constant_j_mol_k
            * step_temperature_k
            / volume_m3
        )
        reference_hydrogen_mol = hydrogen_moles_for_pressure_with_nitrogen(
            target_total_pressure_pa,
            nitrogen_mol=0.0,
            water_vapor_mol=water_state.vapor_mol,
            volume_m3=volume_m3,
            temperature_k=step_temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        reference_hydrogen_partial_pressure_pa = (
            reference_hydrogen_mol
            * gas_constant_j_mol_k
            * step_temperature_k
            / volume_m3
        )
        hydrogen_feedback_factor = hydrogen_partial_pressure_feedback_factor(
            hydrogen_partial_pressure_pa,
            reference_hydrogen_partial_pressure_pa,
            hydrogen_feedback_exponent,
        )
        cell_current_a = base_cell_current_a * hydrogen_feedback_factor
        min_cell_current_a = min(min_cell_current_a, cell_current_a)

        total_pressure = total_gas_pressure_with_nitrogen_pa(
            hydrogen_mol,
            nitrogen_mol,
            water_state.vapor_mol,
            volume_m3=volume_m3,
            temperature_k=step_temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        min_total_pressure_pa = min(min_total_pressure_pa, total_pressure)
        max_total_pressure_pa = max(max_total_pressure_pa, total_pressure)
        max_n2_mole_fraction = max(max_n2_mole_fraction, n2_mole_fraction)
        min_h2_mole_fraction = min(min_h2_mole_fraction, h2_mole_fraction)
        min_hydrogen_feedback_factor = min(
            min_hydrogen_feedback_factor,
            hydrogen_feedback_factor,
        )

        if step % write_every == 0 or step == n_steps:
            rows.append(
                {
                    "regime": regime,
                    "time_s": time_s,
                    "temperature_k": step_temperature_k,
                    "purge_open": purge_remaining_s > 0.0,
                    "purge_count": len(purge_events),
                    "anode_relative_humidity": water_state.relative_humidity,
                    "patch_current_a": patch_current_a,
                    "base_cell_current_a": base_cell_current_a,
                    "hydrogen_feedback_factor": hydrogen_feedback_factor,
                    "cell_current_a": cell_current_a,
                    "charge_since_purge_as": charge_since_purge_as,
                    "hydrogen_mol": hydrogen_mol,
                    "nitrogen_mol": nitrogen_mol,
                    "water_vapor_mol": water_state.vapor_mol,
                    "liquid_water_mol": water_state.liquid_mol,
                    "hydrogen_mole_fraction": h2_mole_fraction,
                    "nitrogen_mole_fraction": n2_mole_fraction,
                    "total_pressure_pa": total_pressure,
                }
            )

        if step == n_steps:
            break

        actual_dt = min(dt_s, stop_time_s - time_s)

        water_source = lambda_flux_to_water_molar_rate(
            flux_lambda,
            membrane_area_m2=active_area_m2,
            fixed_charge_mol_m3=fixed_charge_mol_m3,
        )
        water_state = advance_anode_water_state(
            water_state,
            water_source_mol_s=water_source,
            dt_s=actual_dt,
            volume_m3=volume_m3,
            temperature_k=step_temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        cumulative_water_transfer_mol += water_source * actual_dt

        dynamic_permeance = n2_crossover_permeance_mol_m2_s_pa
        if n2_permeance_model is not None:
            dynamic_permeance = n2_permeance_model(
                water_state.relative_humidity
            )
        elif n2_state_permeance_model is not None:
            current_density_a_m2 = cell_current_a / active_area_m2
            dynamic_permeance = n2_state_permeance_model(
                water_state.relative_humidity,
                current_density_a_m2,
            )
        elif n2_state_temperature_permeance_model is not None:
            current_density_a_m2 = cell_current_a / active_area_m2
            dynamic_permeance = n2_state_temperature_permeance_model(
                water_state.relative_humidity,
                current_density_a_m2,
                step_temperature_k,
            )
        if dynamic_permeance is not None and dynamic_permeance < 0.0:
            raise ValueError("n2 permeance model returned a negative value")

        if dynamic_permeance is None:
            n2_flux = n2_crossover_flux_mol_m2_s
            n2_source_rate = constant_n2_source_rate
        else:
            step_cathode_n2_partial_pressure_pa = (
                cathode_n2_partial_pressure_pa
                if cathode_n2_partial_pressure_model is None
                else float(
                    cathode_n2_partial_pressure_model(step_temperature_k)
                )
            )
            if step_cathode_n2_partial_pressure_pa is None:
                raise RuntimeError(
                    "cathode N2 partial pressure missing in pressure-driven mode"
                )
            if step_cathode_n2_partial_pressure_pa <= 0.0:
                raise ValueError(
                    "cathode N2 partial pressure model returned non-positive value"
                )
            min_n2_permeance = min(min_n2_permeance, dynamic_permeance)
            max_n2_permeance = max(max_n2_permeance, dynamic_permeance)
            anode_n2_partial_pressure_pa = (
                nitrogen_mol
                * gas_constant_j_mol_k
                * step_temperature_k
                / volume_m3
            )
            n2_flux = nitrogen_pressure_driven_flux(
                dynamic_permeance,
                step_cathode_n2_partial_pressure_pa,
                anode_n2_partial_pressure_pa,
            )
            n2_source_rate = nitrogen_crossover_molar_rate(
                n2_flux,
                active_area_m2,
            )
        min_n2_crossover_flux = min(min_n2_crossover_flux, n2_flux)
        max_n2_crossover_flux = max(max_n2_crossover_flux, n2_flux)
        n2_added = n2_source_rate * actual_dt
        nitrogen_mol += n2_added
        cumulative_n2_crossover_mol += n2_added

        consumption_rate = hydrogen_consumption_molar_rate(
            cell_current_a,
            faraday_c_mol,
        )
        consumed_h2 = min(consumption_rate * actual_dt, hydrogen_mol)
        hydrogen_mol -= consumed_h2
        cumulative_h2_consumed_mol += consumed_h2

        if purge_remaining_s > 0.0:
            pressure_before_outflow = total_gas_pressure_with_nitrogen_pa(
                hydrogen_mol,
                nitrogen_mol,
                water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=step_temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            outflow_rate = pressure_driven_purge_molar_rate(
                pressure_before_outflow,
                ambient_pressure_pa,
                purge_conductance,
            )
            gas_state = AnodeGasState(
                hydrogen_mol=hydrogen_mol,
                nitrogen_mol=nitrogen_mol,
                water=water_state,
            )
            gas_state, h2_out, n2_out, water_out = remove_well_mixed_h2_n2_h2o(
                gas_state,
                gas_outflow_mol=outflow_rate * actual_dt,
                volume_m3=volume_m3,
                temperature_k=step_temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            hydrogen_mol = gas_state.hydrogen_mol
            nitrogen_mol = gas_state.nitrogen_mol
            water_state = gas_state.water
            cumulative_h2_purged_mol += h2_out
            cumulative_n2_purged_mol += n2_out
            cumulative_water_purged_mol += water_out

            pressure_after_outflow = total_gas_pressure_with_nitrogen_pa(
                hydrogen_mol,
                nitrogen_mol,
                water_state.vapor_mol,
                volume_m3=volume_m3,
                temperature_k=step_temperature_k,
                gas_constant_j_mol_k=gas_constant_j_mol_k,
            )
            min_total_pressure_pa = min(
                min_total_pressure_pa,
                pressure_after_outflow,
            )

            if active_event is None:
                raise RuntimeError("purge is active without an event record")
            active_event["h2_purged_mol"] += h2_out
            active_event["n2_purged_mol"] += n2_out
            active_event["water_purged_mol"] += water_out
            active_event["min_pressure_pa"] = min(
                active_event["min_pressure_pa"],
                pressure_after_outflow,
            )

            purge_remaining_s = max(purge_remaining_s - actual_dt, 0.0)

        target_hydrogen = hydrogen_moles_for_pressure_with_nitrogen(
            target_total_pressure_pa,
            nitrogen_mol=nitrogen_mol,
            water_vapor_mol=water_state.vapor_mol,
            volume_m3=volume_m3,
            temperature_k=step_temperature_k,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
        )
        refill_h2 = max(target_hydrogen - hydrogen_mol, 0.0)
        hydrogen_mol += refill_h2
        cumulative_h2_inlet_mol += refill_h2

        if active_event is not None:
            active_event["h2_refill_mol"] += refill_h2
            if purge_remaining_s <= 1.0e-12:
                active_event["end_time_s"] = time_s + actual_dt
                active_event["rh_after"] = water_state.relative_humidity
                active_event["n2_after_mol"] = nitrogen_mol
                active_event["pressure_after_pa"] = (
                    total_gas_pressure_with_nitrogen_pa(
                        hydrogen_mol,
                        nitrogen_mol,
                        water_state.vapor_mol,
                        volume_m3=volume_m3,
                        temperature_k=step_temperature_k,
                        gas_constant_j_mol_k=gas_constant_j_mol_k,
                    )
                )
                purge_events.append(active_event)
                active_event = None

        charge_since_purge_as += cell_current_a * actual_dt
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
                "n2_before_mol": nitrogen_mol,
                "n2_after_mol": None,
                "pressure_before_pa": total_gas_pressure_with_nitrogen_pa(
                    hydrogen_mol,
                    nitrogen_mol,
                    water_state.vapor_mol,
                    volume_m3=volume_m3,
                    temperature_k=step_temperature_k,
                    gas_constant_j_mol_k=gas_constant_j_mol_k,
                ),
                "pressure_after_pa": None,
                "min_pressure_pa": float("inf"),
                "h2_purged_mol": 0.0,
                "n2_purged_mol": 0.0,
                "water_purged_mol": 0.0,
                "h2_refill_mol": 0.0,
            }

    final_total_water = water_state.vapor_mol + water_state.liquid_mol
    h2_balance_error = (
        initial_hydrogen_mol
        + cumulative_h2_inlet_mol
        - cumulative_h2_consumed_mol
        - cumulative_h2_purged_mol
        - hydrogen_mol
    )
    n2_balance_error = (
        cumulative_n2_crossover_mol
        - cumulative_n2_purged_mol
        - nitrogen_mol
    )
    water_balance_error = (
        initial_water
        + cumulative_water_transfer_mol
        - cumulative_water_purged_mol
        - final_total_water
    )

    event_times = [float(event["start_time_s"]) for event in purge_events]
    periods = [
        event_times[index] - event_times[index - 1]
        for index in range(1, len(event_times))
    ]

    final_gas_total = hydrogen_mol + nitrogen_mol + water_state.vapor_mol
    summary = {
        "regime": regime,
        "purge_count": len(purge_events),
        "first_purge_time_s": event_times[0] if event_times else None,
        "mean_purge_period_s": mean(periods) if periods else None,
        "purge_duration_s": purge_duration_s,
        "n2_crossover_mode": (
            "state_dependent_permeance"
            if n2_permeance_model is not None
            else (
                "constant_flux"
                if n2_crossover_permeance_mol_m2_s_pa is None
                else "partial_pressure_driven"
            )
        ),
        "n2_crossover_flux_mol_m2_s": n2_crossover_flux_mol_m2_s,
        "n2_crossover_permeance_mol_m2_s_pa": (
            n2_crossover_permeance_mol_m2_s_pa
        ),
        "cathode_n2_partial_pressure_pa": cathode_n2_partial_pressure_pa,
        "n2_crossover_rate_mol_s": (
            cumulative_n2_crossover_mol / stop_time_s
        ),
        "mean_n2_crossover_rate_mol_s": (
            cumulative_n2_crossover_mol / stop_time_s
        ),
        "mean_n2_crossover_flux_mol_m2_s": (
            cumulative_n2_crossover_mol / (active_area_m2 * stop_time_s)
        ),
        "min_n2_crossover_flux_mol_m2_s": (
            0.0
            if min_n2_crossover_flux == float("inf")
            else min_n2_crossover_flux
        ),
        "max_n2_crossover_flux_mol_m2_s": max_n2_crossover_flux,
        "min_n2_permeance_mol_m2_s_pa": (
            None
            if min_n2_permeance == float("inf")
            else min_n2_permeance
        ),
        "max_n2_permeance_mol_m2_s_pa": (
            None if max_n2_permeance == 0.0 else max_n2_permeance
        ),
        "hydrogen_feedback_exponent": hydrogen_feedback_exponent,
        "min_hydrogen_feedback_factor": min_hydrogen_feedback_factor,
        "min_cell_current_a": min_cell_current_a,
        "mean_cell_current_a": (
            2.0
            * faraday_c_mol
            * cumulative_h2_consumed_mol
            / stop_time_s
        ),
        "final_relative_humidity": water_state.relative_humidity,
        "final_hydrogen_mol": hydrogen_mol,
        "final_nitrogen_mol": nitrogen_mol,
        "final_hydrogen_mole_fraction": (
            hydrogen_mol / final_gas_total if final_gas_total > 0.0 else 0.0
        ),
        "final_nitrogen_mole_fraction": (
            nitrogen_mol / final_gas_total if final_gas_total > 0.0 else 0.0
        ),
        "max_nitrogen_mole_fraction": max_n2_mole_fraction,
        "min_hydrogen_mole_fraction": min_h2_mole_fraction,
        "min_total_pressure_pa": min_total_pressure_pa,
        "max_total_pressure_pa": max_total_pressure_pa,
        "cumulative_h2_inlet_mol": cumulative_h2_inlet_mol,
        "cumulative_h2_consumed_mol": cumulative_h2_consumed_mol,
        "cumulative_h2_purged_mol": cumulative_h2_purged_mol,
        "cumulative_n2_crossover_mol": cumulative_n2_crossover_mol,
        "cumulative_n2_purged_mol": cumulative_n2_purged_mol,
        "cumulative_water_transfer_mol": cumulative_water_transfer_mol,
        "cumulative_water_purged_mol": cumulative_water_purged_mol,
        "h2_balance_error_mol": h2_balance_error,
        "n2_balance_error_mol": n2_balance_error,
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
    parser.add_argument("--n2-crossover-flux", type=float, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-anode-nitrogen.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-anode-nitrogen.csv"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.initial_rh <= 1.0:
        parser.error("--initial-rh must be in [0, 1]")
    if args.stop_time <= 0.0 or args.dt <= 0.0:
        parser.error("--stop-time and --dt must be positive")
    if args.write_every < 1:
        parser.error("--write-every must be >= 1")

    p = CathodeParameters()
    n2_flux = (
        p.n2_crossover_flux_mol_m2_s
        if args.n2_crossover_flux is None
        else args.n2_crossover_flux
    )
    if n2_flux < 0.0:
        parser.error("--n2-crossover-flux must be non-negative")

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

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for regime in args.regimes:
        rows, summary = simulate_nitrogen_regime(
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
            n2_crossover_flux_mol_m2_s=n2_flux,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{regime}: purges={summary['purge_count']} "
            f"xN2,max={summary['max_nitrogen_mole_fraction']:.5f} "
            f"xH2,min={summary['min_hydrogen_mole_fraction']:.5f} "
            f"N2bal={summary['n2_balance_error_mol']:.3e}",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-anode-nitrogen",
        "closure_source": str(args.closure_csv),
        "area_scale_factor": scaling.area_scale_factor,
        "inferred_active_area_cm2": scaling.inferred_active_area_cm2,
        "n2_crossover_flux_mol_m2_s": n2_flux,
        "purge_duration_s": p.tech.purge_duration_max_s,
        "purge_reference_flow_slpm": p.tech.purge_rate_min_slpm_per_cell,
        "purge_interval_as": p.tech.purge_interval_as,
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
