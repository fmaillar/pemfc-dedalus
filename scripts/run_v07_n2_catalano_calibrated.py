"""V0.7 Catalano-calibrated N2 permeability study.

This campaign makes the water-volume closure more internally consistent with
Catalano et al. (2012):

- dry Nafion 117 N2 permeability: 0.18 Barrer at 35 C;
- dry N2 permeation activation energy: 49.6 kJ/mol;
- N2 permeability increase at water activity 0.75: about 95x;
- maximum humidity enhancement capped at 100x.

The water volume fraction uses a partial molar water volume. Two values are
screened:

- 17.0 cm3/mol, a literature estimate for hydrated Nafion;
- 18.015 cm3/mol, bulk-water volume additivity.

The exponential slope is calibrated so that a membrane uniformly equilibrated
at activity 0.75 reaches a 95x permeability ratio in the model's own
lambda-to-volume conversion.
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
    calibrated_exponential_permeability_multiplier,
    membrane_permeance_from_permeability_profile,
    membrane_water_volume_fraction_from_partial_molar_volume,
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

DRY_REFERENCE_PERMEABILITY_BARRER = 0.18
REFERENCE_TEMPERATURE_K = 308.15
DRY_N2_ACTIVATION_ENERGY_J_MOL = 49_600.0
HYDRATED_N2_ACTIVATION_ENERGY_J_MOL = 19_830.0
ANCHOR_WATER_ACTIVITY = 0.75
ANCHOR_PERMEABILITY_FACTOR = 95.0
MAXIMUM_PERMEABILITY_FACTOR = 100.0
DEFAULT_WATER_PARTIAL_MOLAR_VOLUMES_CM3_MOL = [17.0, 18.01528]
DEFAULT_ACTIVATION_ENERGIES_J_MOL = [
    DRY_N2_ACTIVATION_ENERGY_J_MOL,
    HYDRATED_N2_ACTIVATION_ENERGY_J_MOL,
]
DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]
MEMBRANE_PROFILE_POINTS = 65


def scalar_water_content(activity: float) -> float:
    """Return scalar Nafion water content for scalar water activity."""
    return float(membrane_water_content_from_activity(activity))


def make_calibrated_permeance_model(
    *,
    dry_reference_si: float,
    stack_temperature_k: float,
    gas_constant_j_mol_k: float,
    activation_energy_j_mol: float,
    cathode_relative_humidity: float,
    membrane_thickness_m: float,
    membrane_equivalent_weight_kg_mol: float,
    membrane_dry_density_kg_m3: float,
    water_partial_molar_volume_m3_mol: float,
) -> tuple[Callable[[float], float], dict[str, float]]:
    """Build the Catalano-calibrated state-dependent permeance closure."""
    lambda_dry = scalar_water_content(0.0)
    lambda_anchor = scalar_water_content(ANCHOR_WATER_ACTIVITY)
    lambda_cathode = scalar_water_content(cathode_relative_humidity)

    phi_dry = membrane_water_volume_fraction_from_partial_molar_volume(
        lambda_dry,
        membrane_equivalent_weight_kg_mol=membrane_equivalent_weight_kg_mol,
        membrane_dry_density_kg_m3=membrane_dry_density_kg_m3,
        water_partial_molar_volume_m3_mol=water_partial_molar_volume_m3_mol,
    )
    phi_anchor = membrane_water_volume_fraction_from_partial_molar_volume(
        lambda_anchor,
        membrane_equivalent_weight_kg_mol=membrane_equivalent_weight_kg_mol,
        membrane_dry_density_kg_m3=membrane_dry_density_kg_m3,
        water_partial_molar_volume_m3_mol=water_partial_molar_volume_m3_mol,
    )

    def permeance_model(anode_rh: float) -> float:
        lambda_anode = scalar_water_content(anode_rh)
        dry_at_temperature = arrhenius_permeability(
            dry_reference_si,
            stack_temperature_k,
            REFERENCE_TEMPERATURE_K,
            activation_energy_j_mol,
            gas_constant_j_mol_k,
        )

        permeability_profile: list[float] = []
        for index in range(MEMBRANE_PROFILE_POINTS):
            fraction = index / (MEMBRANE_PROFILE_POINTS - 1)
            lambda_local = lambda_anode + fraction * (
                lambda_cathode - lambda_anode
            )
            phi_water = membrane_water_volume_fraction_from_partial_molar_volume(
                lambda_local,
                membrane_equivalent_weight_kg_mol=(
                    membrane_equivalent_weight_kg_mol
                ),
                membrane_dry_density_kg_m3=membrane_dry_density_kg_m3,
                water_partial_molar_volume_m3_mol=(
                    water_partial_molar_volume_m3_mol
                ),
            )
            humidity_factor = calibrated_exponential_permeability_multiplier(
                phi_water,
                baseline_fraction=phi_dry,
                anchor_fraction=phi_anchor,
                anchor_factor=ANCHOR_PERMEABILITY_FACTOR,
                maximum_factor=MAXIMUM_PERMEABILITY_FACTOR,
            )
            permeability_profile.append(
                dry_at_temperature * humidity_factor
            )

        return membrane_permeance_from_permeability_profile(
            permeability_profile,
            membrane_thickness_m,
        )

    metadata = {
        "lambda_dry": lambda_dry,
        "lambda_anchor": lambda_anchor,
        "lambda_cathode": lambda_cathode,
        "phi_dry": phi_dry,
        "phi_anchor": phi_anchor,
    }
    return permeance_model, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty Catalano N2 table")
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
        "--water-partial-molar-volumes-cm3-mol",
        nargs="+",
        type=float,
        default=DEFAULT_WATER_PARTIAL_MOLAR_VOLUMES_CM3_MOL,
    )
    parser.add_argument(
        "--activation-energies-j-mol",
        nargs="+",
        type=float,
        default=DEFAULT_ACTIVATION_ENERGIES_J_MOL,
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
        default=Path("results/v07-n2-catalano-calibrated.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-n2-catalano-calibrated.csv"),
    )
    args = parser.parse_args()

    if not args.water_partial_molar_volumes_cm3_mol:
        parser.error("--water-partial-molar-volumes-cm3-mol must not be empty")
    if any(value <= 0.0 for value in args.water_partial_molar_volumes_cm3_mol):
        parser.error("water partial molar volumes must be positive")
    if not args.activation_energies_j_mol:
        parser.error("--activation-energies-j-mol must not be empty")
    if any(value < 0.0 for value in args.activation_energies_j_mol):
        parser.error("activation energies must be non-negative")
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
    rows: list[dict[str, Any]] = []

    for water_volume_cm3_mol in args.water_partial_molar_volumes_cm3_mol:
        water_volume_m3_mol = water_volume_cm3_mol * 1.0e-6
        for activation_energy in args.activation_energies_j_mol:
            permeance_model, anchors = make_calibrated_permeance_model(
                dry_reference_si=dry_reference_si,
                stack_temperature_k=p.stack_temperature,
                gas_constant_j_mol_k=p.gas_constant,
                activation_energy_j_mol=activation_energy,
                cathode_relative_humidity=p.relative_humidity,
                membrane_thickness_m=p.membrane_thickness,
                membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
                membrane_dry_density_kg_m3=p.membrane_dry_density,
                water_partial_molar_volume_m3_mol=water_volume_m3_mol,
            )

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
                        purge_reference_flow_slpm=(
                            p.tech.purge_rate_min_slpm_per_cell
                        ),
                        current_scale_factor=scaling.area_scale_factor,
                        n2_crossover_flux_mol_m2_s=0.0,
                        hydrogen_feedback_exponent=feedback_exponent,
                        cathode_n2_partial_pressure_pa=(
                            cathode_n2_partial_pressure
                        ),
                        n2_permeance_model=permeance_model,
                    )

                    final_lambda_anode = scalar_water_content(
                        float(summary["final_relative_humidity"])
                    )
                    final_lambda_effective = 0.5 * (
                        final_lambda_anode + anchors["lambda_cathode"]
                    )
                    final_phi = (
                        membrane_water_volume_fraction_from_partial_molar_volume(
                            final_lambda_effective,
                            membrane_equivalent_weight_kg_mol=(
                                p.membrane_equivalent_weight
                            ),
                            membrane_dry_density_kg_m3=p.membrane_dry_density,
                            water_partial_molar_volume_m3_mol=(
                                water_volume_m3_mol
                            ),
                        )
                    )
                    row = {
                        "regime": regime,
                        "hydrogen_feedback_exponent": feedback_exponent,
                        "water_partial_molar_volume_cm3_mol": (
                            water_volume_cm3_mol
                        ),
                        "activation_energy_j_mol": activation_energy,
                        "dry_reference_permeability_barrer": (
                            DRY_REFERENCE_PERMEABILITY_BARRER
                        ),
                        "anchor_water_activity": ANCHOR_WATER_ACTIVITY,
                        "anchor_permeability_factor": (
                            ANCHOR_PERMEABILITY_FACTOR
                        ),
                        "phi_dry": anchors["phi_dry"],
                        "phi_anchor": anchors["phi_anchor"],
                        "final_water_volume_fraction": final_phi,
                        "mean_n2_crossover_flux_mol_m2_s": (
                            summary["mean_n2_crossover_flux_mol_m2_s"]
                        ),
                        "max_nitrogen_mole_fraction": (
                            summary["max_nitrogen_mole_fraction"]
                        ),
                        "min_hydrogen_mole_fraction": (
                            summary["min_hydrogen_mole_fraction"]
                        ),
                        "mean_cell_current_a": summary["mean_cell_current_a"],
                        "min_hydrogen_feedback_factor": (
                            summary["min_hydrogen_feedback_factor"]
                        ),
                        "purge_count": summary["purge_count"],
                        "mean_purge_period_s": summary["mean_purge_period_s"],
                        "h2_balance_error_mol": summary["h2_balance_error_mol"],
                        "n2_balance_error_mol": summary["n2_balance_error_mol"],
                        "water_balance_error_mol": (
                            summary["water_balance_error_mol"]
                        ),
                    }
                    rows.append(row)
                    print(
                        f"{regime} Vw={water_volume_cm3_mol:.3f} "
                        f"Ea={activation_energy / 1000.0:.2f} "
                        f"gamma={feedback_exponent:.1f}: "
                        f"Jmean={row['mean_n2_crossover_flux_mol_m2_s']:.3e} "
                        f"xN2,max={row['max_nitrogen_mole_fraction']:.5f}",
                        flush=True,
                    )

    output = {
        "schema_version": 1,
        "model": "v07-n2-catalano-calibrated",
        "closure_source": str(args.closure_csv),
        "dry_reference_permeability_barrer": (
            DRY_REFERENCE_PERMEABILITY_BARRER
        ),
        "dry_reference_temperature_k": REFERENCE_TEMPERATURE_K,
        "anchor_water_activity": ANCHOR_WATER_ACTIVITY,
        "anchor_permeability_factor": ANCHOR_PERMEABILITY_FACTOR,
        "maximum_permeability_factor": MAXIMUM_PERMEABILITY_FACTOR,
        "water_partial_molar_volumes_cm3_mol": (
            args.water_partial_molar_volumes_cm3_mol
        ),
        "activation_energies_j_mol": args.activation_energies_j_mol,
        "feedback_exponents": args.feedback_exponents,
        "membrane_permeance_model": "through-plane resistance integration",
        "membrane_water_profile": "linear lambda between anode and cathode",
        "membrane_profile_points": MEMBRANE_PROFILE_POINTS,
        "membrane_thickness_m": p.membrane_thickness,
        "stack_temperature_k": p.stack_temperature,
        "cathode_n2_partial_pressure_pa": cathode_n2_partial_pressure,
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
