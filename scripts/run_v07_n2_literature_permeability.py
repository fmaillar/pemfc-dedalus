"""V0.7 literature-anchored N2 membrane-permeability sensitivity.

The study anchors the pressure-driven N2 crossover law to published Nafion
permeability data rather than to an imposed crossover flux.

Dry Nafion H+ at 35 degC has a reported N2 permeability near 0.24 Barrer.
Published PFSI measurements also report humidity enhancements up to about
100x.  This campaign therefore screens:

    0.24, 2.4, 24 Barrer

as dry, 10x-humid and 100x-humid permeability levels.

For the provisional 50 um membrane:

    K_N2 = P_N2 / L_membrane

and the existing pressure-gradient law is then used directly.
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
    barrer_to_si_permeability,
    membrane_permeance_from_permeability,
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


DEFAULT_PERMEABILITIES_BARRER = [0.24, 2.4, 24.0]
DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty literature permeability table")
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
        "--permeabilities-barrer",
        nargs="+",
        type=float,
        default=DEFAULT_PERMEABILITIES_BARRER,
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
        default=Path("results/v07-n2-literature-permeability.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-n2-literature-permeability.csv"),
    )
    args = parser.parse_args()

    if not args.permeabilities_barrer:
        parser.error("--permeabilities-barrer must not be empty")
    if any(value < 0.0 for value in args.permeabilities_barrer):
        parser.error("--permeabilities-barrer must be non-negative")
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

    rows: list[dict[str, Any]] = []

    for permeability_barrer in args.permeabilities_barrer:
        permeability_si = barrer_to_si_permeability(permeability_barrer)
        permeance = membrane_permeance_from_permeability(
            permeability_si,
            p.membrane_thickness,
        )
        initial_flux = permeance * cathode_n2_partial_pressure

        for exponent in args.feedback_exponents:
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
                    n2_crossover_flux_mol_m2_s=initial_flux,
                    hydrogen_feedback_exponent=exponent,
                    n2_crossover_permeance_mol_m2_s_pa=permeance,
                    cathode_n2_partial_pressure_pa=cathode_n2_partial_pressure,
                )

                row = {
                    "regime": regime,
                    "n2_permeability_barrer": permeability_barrer,
                    "n2_permeability_si_mol_m_per_m2_s_pa": permeability_si,
                    "membrane_thickness_m": p.membrane_thickness,
                    "n2_permeance_mol_m2_s_pa": permeance,
                    "cathode_n2_partial_pressure_pa": (
                        cathode_n2_partial_pressure
                    ),
                    "initial_n2_flux_mol_m2_s": initial_flux,
                    "hydrogen_feedback_exponent": exponent,
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
                    "final_nitrogen_mole_fraction": (
                        summary["final_nitrogen_mole_fraction"]
                    ),
                    "cumulative_n2_crossover_mol": (
                        summary["cumulative_n2_crossover_mol"]
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
                    f"{regime} P_N2={permeability_barrer:.2f} Barrer "
                    f"gamma={exponent:.1f}: "
                    f"J0={initial_flux:.3e} "
                    f"xN2,max={row['max_nitrogen_mole_fraction']:.5f}",
                    flush=True,
                )

    output = {
        "schema_version": 1,
        "model": "v07-n2-literature-permeability",
        "closure_source": str(args.closure_csv),
        "literature_basis": {
            "dry_nafion_hplus_n2_permeability_barrer_at_35c": 0.24,
            "reported_humidity_enhancement_upper_factor": 100.0,
            "screened_permeabilities_barrer": args.permeabilities_barrer,
        },
        "crossover_law": (
            "(P_N2 / membrane_thickness) * "
            "max(p_N2_cathode - p_N2_anode, 0)"
        ),
        "membrane_thickness_m": p.membrane_thickness,
        "cathode_n2_partial_pressure_pa": cathode_n2_partial_pressure,
        "feedback_exponents": args.feedback_exponents,
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
