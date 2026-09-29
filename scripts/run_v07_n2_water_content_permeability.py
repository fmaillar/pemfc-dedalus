"""V0.7 N2 permeability driven by membrane water content.

Catalano et al. report that gas permeability in PFSI membranes increases
approximately exponentially with membrane water volume fraction over the
measured hydrated range. This study replaces the arbitrary RH-shape exponent
with a water-content proxy based on the existing Nafion lambda(a) relation.

The closure is

    lambda_eff = (lambda_anode + lambda_cathode) / 2

and

    F_water = 100 ** xi

where xi maps lambda_eff between the dry lambda value and the lambda value at
water activity 0.9. Temperature scaling remains Arrhenius.

This is still a screening closure because lambda is used as a proxy for water
volume fraction; it is not a digitized fit of the Catalano data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
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
    water_content_dependent_permeability,
)
from pemfc_dedalus.membrane import (
    membrane_fixed_charge_concentration,
    membrane_water_content_from_activity,
)
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
REFERENCE_WATER_ACTIVITY = 0.90
DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]


def scalar_water_content(activity: float) -> float:
    """Return scalar Nafion water content for a scalar water activity."""
    return float(membrane_water_content_from_activity(activity))


def make_water_content_permeance_model(
    *,
    dry_reference_si: float,
    stack_temperature_k: float,
    gas_constant_j_mol_k: float,
    cathode_relative_humidity: float,
    membrane_thickness_m: float,
) -> tuple[Callable[[float], float], float, float, float]:
    """Build the dynamic permeance closure and return its lambda anchors."""
    lambda_dry = scalar_water_content(0.0)
    lambda_reference = scalar_water_content(REFERENCE_WATER_ACTIVITY)
    lambda_cathode = scalar_water_content(cathode_relative_humidity)

    def permeance_model(anode_rh: float) -> float:
        lambda_anode = scalar_water_content(anode_rh)
        lambda_effective = 0.5 * (lambda_anode + lambda_cathode)
        permeability = water_content_dependent_permeability(
            dry_reference_si,
            temperature_k=stack_temperature_k,
            reference_temperature_k=REFERENCE_TEMPERATURE_K,
            activation_energy_j_mol=ACTIVATION_ENERGY_J_MOL,
            gas_constant_j_mol_k=gas_constant_j_mol_k,
            water_content=lambda_effective,
            dry_water_content=lambda_dry,
            reference_water_content=lambda_reference,
            maximum_humidity_factor=MAXIMUM_HUMIDITY_FACTOR,
        )
        return membrane_permeance_from_permeability(
            permeability,
            membrane_thickness_m,
        )

    return permeance_model, lambda_dry, lambda_reference, lambda_cathode


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty water-content N2 table")
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
        default=Path("results/v07-n2-water-content-permeability.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-n2-water-content-permeability.csv"),
    )
    args = parser.parse_args()

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
    (
        permeance_model,
        lambda_dry,
        lambda_reference,
        lambda_cathode,
    ) = make_water_content_permeance_model(
        dry_reference_si=dry_reference_si,
        stack_temperature_k=p.stack_temperature,
        gas_constant_j_mol_k=p.gas_constant,
        cathode_relative_humidity=p.relative_humidity,
        membrane_thickness_m=p.membrane_thickness,
    )

    rows: list[dict[str, Any]] = []

    for feedback_exponent in args.feedback_exponents:
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
                n2_crossover_flux_mol_m2_s=0.0,
                hydrogen_feedback_exponent=feedback_exponent,
                cathode_n2_partial_pressure_pa=cathode_n2_partial_pressure,
                n2_permeance_model=permeance_model,
            )

            final_lambda_anode = scalar_water_content(
                float(summary["final_relative_humidity"])
            )
            final_lambda_effective = 0.5 * (
                final_lambda_anode + lambda_cathode
            )
            row = {
                "regime": regime,
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
                "lambda_dry": lambda_dry,
                "lambda_reference_at_a09": lambda_reference,
                "lambda_cathode": lambda_cathode,
                "final_lambda_anode": final_lambda_anode,
                "final_lambda_effective": final_lambda_effective,
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
                f"{regime} gamma={feedback_exponent:.1f}: "
                f"lambda_eff={final_lambda_effective:.3f} "
                f"Jmean={row['mean_n2_crossover_flux_mol_m2_s']:.3e} "
                f"xN2,max={row['max_nitrogen_mole_fraction']:.5f}",
                flush=True,
            )

    output = {
        "schema_version": 1,
        "model": "v07-n2-water-content-permeability",
        "closure_source": str(args.closure_csv),
        "literature_interpretation": (
            "exponential permeability increase represented versus Nafion "
            "water content lambda as a proxy for water volume fraction"
        ),
        "dry_reference_permeability_barrer": (
            DRY_REFERENCE_PERMEABILITY_BARRER
        ),
        "dry_reference_temperature_k": REFERENCE_TEMPERATURE_K,
        "temperature_law": "Arrhenius",
        "activation_energy_j_mol": ACTIVATION_ENERGY_J_MOL,
        "water_content_law": (
            "100 ** clip((lambda_eff-lambda_dry)/"
            "(lambda_a0.9-lambda_dry), 0, 1)"
        ),
        "effective_water_content": "(lambda_anode + lambda_cathode) / 2",
        "lambda_dry": lambda_dry,
        "lambda_reference_at_a09": lambda_reference,
        "lambda_cathode": lambda_cathode,
        "feedback_exponents": args.feedback_exponents,
        "membrane_thickness_m": p.membrane_thickness,
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
