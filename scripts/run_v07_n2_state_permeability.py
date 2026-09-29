"""V0.7 state-dependent N2 permeability sensitivity.

This study replaces the discrete dry/10x/100x permeability levels with a
continuous screening closure that depends on membrane temperature and an
effective membrane relative humidity.

Temperature scaling uses an Arrhenius law with the published N2 activation
energy of 19.83 kJ/mol from fully hydrated PEMFC measurements.

Humidity scaling is deliberately empirical. Published PFSI data show strongly
nonlinear N2 permeability enhancement, up to about 100x by high RH, but do not
define one universal closed-form law. The model therefore interpolates between
1x at dry conditions and 100x at RH=0.9 using a screened shape exponent.

The effective membrane RH is the arithmetic mean of the dynamic anode RH and
the fixed cathode RH. This is a first-order closure, not a resolved membrane
water-activity profile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.anode_nitrogen import (
    humid_air_nitrogen_partial_pressure_pa,
)
from pemfc_dedalus.gas_permeability import (
    arrhenius_permeability,
    barrer_to_si_permeability,
    membrane_permeance_from_permeability,
    state_dependent_permeability,
)
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime

DRY_REFERENCE_PERMEABILITY_BARRER = 0.24
REFERENCE_TEMPERATURE_K = 308.15
ACTIVATION_ENERGY_J_MOL = 19_830.0
MAXIMUM_HUMIDITY_FACTOR = 100.0
HUMIDITY_REFERENCE_RH = 0.90
DEFAULT_HUMIDITY_SHAPE_EXPONENTS = [1.0, 2.0, 4.0]
DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty state-dependent N2 table")
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
        "--humidity-shape-exponents",
        nargs="+",
        type=float,
        default=DEFAULT_HUMIDITY_SHAPE_EXPONENTS,
    )
    parser.add_argument(
        "--feedback-exponents",
        nargs="+",
        type=float,
        default=DEFAULT_FEEDBACK_EXPONENTS,
    )
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-n2-state-permeability.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-n2-state-permeability.csv"),
    )
    args = parser.parse_args()

    if not args.humidity_shape_exponents:
        parser.error("--humidity-shape-exponents must not be empty")
    if any(value <= 0.0 for value in args.humidity_shape_exponents):
        parser.error("--humidity-shape-exponents must be positive")
    if not args.feedback_exponents:
        parser.error("--feedback-exponents must not be empty")
    if any(value < 0.0 for value in args.feedback_exponents):
        parser.error("--feedback-exponents must be non-negative")
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

    saturation_pressure = water_saturation_pressure_pa(p.stack_temperature)
    cathode_n2_partial_pressure = humid_air_nitrogen_partial_pressure_pa(
        total_pressure_pa=p.pressure,
        oxygen_dry_mole_fraction=p.oxygen_mole_fraction,
        relative_humidity=p.relative_humidity,
        saturation_water_pressure_pa=saturation_pressure,
    )

    dry_reference_si = barrer_to_si_permeability(
        DRY_REFERENCE_PERMEABILITY_BARRER
    )
    dry_at_stack_temperature = arrhenius_permeability(
        dry_reference_si,
        p.stack_temperature,
        REFERENCE_TEMPERATURE_K,
        ACTIVATION_ENERGY_J_MOL,
        p.gas_constant,
    )

    rows: list[dict[str, Any]] = []

    for shape_exponent in args.humidity_shape_exponents:
        for feedback_exponent in args.feedback_exponents:
            for regime in args.regimes:

                def permeance_model(anode_rh: float) -> float:
                    permeability = state_dependent_permeability(
                        dry_reference_si,
                        temperature_k=p.stack_temperature,
                        reference_temperature_k=REFERENCE_TEMPERATURE_K,
                        activation_energy_j_mol=ACTIVATION_ENERGY_J_MOL,
                        gas_constant_j_mol_k=p.gas_constant,
                        anode_relative_humidity=anode_rh,
                        cathode_relative_humidity=p.relative_humidity,
                        maximum_humidity_factor=MAXIMUM_HUMIDITY_FACTOR,
                        humidity_reference_relative_humidity=HUMIDITY_REFERENCE_RH,
                        humidity_shape_exponent=shape_exponent,
                    )
                    return membrane_permeance_from_permeability(
                        permeability,
                        p.membrane_thickness,
                    )

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
                    n2_crossover_flux_mol_m2_s=0.0,
                    hydrogen_feedback_exponent=feedback_exponent,
                    cathode_n2_partial_pressure_pa=cathode_n2_partial_pressure,
                    n2_permeance_model=permeance_model,
                )

                row = {
                    "regime": regime,
                    "humidity_shape_exponent": shape_exponent,
                    "hydrogen_feedback_exponent": feedback_exponent,
                    "dry_reference_permeability_barrer": (
                        DRY_REFERENCE_PERMEABILITY_BARRER
                    ),
                    "reference_temperature_k": REFERENCE_TEMPERATURE_K,
                    "stack_temperature_k": p.stack_temperature,
                    "activation_energy_j_mol": ACTIVATION_ENERGY_J_MOL,
                    "dry_permeability_at_stack_temperature_si": (
                        dry_at_stack_temperature
                    ),
                    "cathode_relative_humidity": p.relative_humidity,
                    "maximum_humidity_factor": MAXIMUM_HUMIDITY_FACTOR,
                    "humidity_reference_relative_humidity": (
                        HUMIDITY_REFERENCE_RH
                    ),
                    "min_n2_permeance_mol_m2_s_pa": (
                        summary["min_n2_permeance_mol_m2_s_pa"]
                    ),
                    "max_n2_permeance_mol_m2_s_pa": (
                        summary["max_n2_permeance_mol_m2_s_pa"]
                    ),
                    "mean_n2_crossover_flux_mol_m2_s": (
                        summary["mean_n2_crossover_flux_mol_m2_s"]
                    ),
                    "min_n2_crossover_flux_mol_m2_s": (
                        summary["min_n2_crossover_flux_mol_m2_s"]
                    ),
                    "max_n2_crossover_flux_mol_m2_s": (
                        summary["max_n2_crossover_flux_mol_m2_s"]
                    ),
                    "mean_cell_current_a": summary["mean_cell_current_a"],
                    "min_hydrogen_feedback_factor": (
                        summary["min_hydrogen_feedback_factor"]
                    ),
                    "purge_count": summary["purge_count"],
                    "mean_purge_period_s": summary["mean_purge_period_s"],
                    "max_nitrogen_mole_fraction": (
                        summary["max_nitrogen_mole_fraction"]
                    ),
                    "min_hydrogen_mole_fraction": (
                        summary["min_hydrogen_mole_fraction"]
                    ),
                    "final_relative_humidity": summary["final_relative_humidity"],
                    "h2_balance_error_mol": summary["h2_balance_error_mol"],
                    "n2_balance_error_mol": summary["n2_balance_error_mol"],
                    "water_balance_error_mol": summary["water_balance_error_mol"],
                }
                rows.append(row)
                print(
                    f"{regime} q_RH={shape_exponent:.1f} "
                    f"gamma={feedback_exponent:.1f}: "
                    f"Jmean={row['mean_n2_crossover_flux_mol_m2_s']:.3e} "
                    f"xN2,max={row['max_nitrogen_mole_fraction']:.5f}",
                    flush=True,
                )

    output = {
        "schema_version": 1,
        "model": "v07-n2-state-permeability",
        "closure_source": str(args.closure_csv),
        "dry_reference_permeability_barrer": (
            DRY_REFERENCE_PERMEABILITY_BARRER
        ),
        "dry_reference_temperature_k": REFERENCE_TEMPERATURE_K,
        "temperature_law": "Arrhenius",
        "activation_energy_j_mol": ACTIVATION_ENERGY_J_MOL,
        "humidity_law": (
            "100 ** (min(RH_eff / 0.9, 1) ** q_RH)"
        ),
        "effective_membrane_rh": "(RH_anode + RH_cathode) / 2",
        "humidity_shape_exponents": args.humidity_shape_exponents,
        "feedback_exponents": args.feedback_exponents,
        "membrane_thickness_m": p.membrane_thickness,
        "cathode_relative_humidity": p.relative_humidity,
        "cathode_n2_partial_pressure_pa": cathode_n2_partial_pressure,
        "stack_temperature_k": p.stack_temperature,
        "inferred_active_area_cm2": scaling.inferred_active_area_cm2,
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
