"""Compare linear and V0.6 membrane-water profiles for Catalano N2 crossover.

The Catalano permeability law and all anode dynamics are identical between the
two cases. Only the membrane water-content profile differs:

- linear_profile: lambda is linearly interpolated between gas-equilibrium
  anode and cathode boundary values;
- v06_finite_transfer_profile: lambda is computed with the existing V0.6
  steady diffusion + electro-osmotic-drag + finite anode-transfer model.

This isolates the effect of membrane water transport on through-plane N2
permeance before introducing any additional empirical closure.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

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
    electro_osmotic_lambda_velocity,
    membrane_fixed_charge_concentration,
    membrane_water_content_from_activity,
    steady_membrane_water_profile_anode_transfer,
)
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime
from scripts.run_v07_n2_catalano_calibrated import (
    ANCHOR_PERMEABILITY_FACTOR,
    ANCHOR_WATER_ACTIVITY,
    DRY_REFERENCE_PERMEABILITY_BARRER,
    MAXIMUM_PERMEABILITY_FACTOR,
    MEMBRANE_PROFILE_POINTS,
    REFERENCE_TEMPERATURE_K,
    make_calibrated_permeance_model,
)

DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL = 17.0
DEFAULT_ACTIVATION_ENERGY_J_MOL = 49_600.0
DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]


def scalar_water_content(activity: float) -> float:
    """Return scalar Nafion water content for scalar water activity."""
    return float(membrane_water_content_from_activity(activity))


def make_v06_transport_permeance_model(
    *,
    dry_reference_si: float,
    stack_temperature_k: float,
    gas_constant_j_mol_k: float,
    faraday_c_mol: float,
    fixed_charge_mol_m3: float,
    activation_energy_j_mol: float,
    cathode_relative_humidity: float,
    membrane_thickness_m: float,
    membrane_equivalent_weight_kg_mol: float,
    membrane_dry_density_kg_m3: float,
    membrane_water_diffusivity_m2_s: float,
    anode_transfer_coefficient_m_s: float,
    water_partial_molar_volume_m3_mol: float,
    profile_points: int = MEMBRANE_PROFILE_POINTS,
) -> Callable[[float, float], float]:
    """Build Catalano permeance on the V0.6 finite-transfer lambda profile."""
    if profile_points < 2:
        raise ValueError("profile_points must be >= 2")

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
    dry_at_temperature = arrhenius_permeability(
        dry_reference_si,
        stack_temperature_k,
        REFERENCE_TEMPERATURE_K,
        activation_energy_j_mol,
        gas_constant_j_mol_k,
    )
    z_membrane = np.linspace(0.0, membrane_thickness_m, profile_points)

    def permeance_model(
        anode_relative_humidity: float,
        current_density_a_m2: float,
    ) -> float:
        lambda_anode_equilibrium = scalar_water_content(
            anode_relative_humidity
        )
        drag_velocity = electro_osmotic_lambda_velocity(
            current_density_a_m2,
            faraday_c_mol,
            fixed_charge_mol_m3,
        )
        lambda_profile = steady_membrane_water_profile_anode_transfer(
            z_membrane,
            lambda_cathode=lambda_cathode,
            lambda_anode_equilibrium=lambda_anode_equilibrium,
            diffusivity_m2_s=membrane_water_diffusivity_m2_s,
            drag_velocity_m_s=drag_velocity,
            anode_transfer_coefficient_m_s=anode_transfer_coefficient_m_s,
        )

        permeability_profile: list[float] = []
        for lambda_local in lambda_profile:
            phi_water = (
                membrane_water_volume_fraction_from_partial_molar_volume(
                    float(lambda_local),
                    membrane_equivalent_weight_kg_mol=(
                        membrane_equivalent_weight_kg_mol
                    ),
                    membrane_dry_density_kg_m3=membrane_dry_density_kg_m3,
                    water_partial_molar_volume_m3_mol=(
                        water_partial_molar_volume_m3_mol
                    ),
                )
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

    return permeance_model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty membrane-profile comparison")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_change(new: float, reference: float) -> float:
    """Return new/reference - 1."""
    if reference == 0.0:
        raise ValueError("reference value must be non-zero")
    return new / reference - 1.0


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
    parser.add_argument(
        "--water-partial-molar-volume-cm3-mol",
        type=float,
        default=DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
    )
    parser.add_argument(
        "--activation-energy-j-mol",
        type=float,
        default=DEFAULT_ACTIVATION_ENERGY_J_MOL,
    )
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument("--stop-time", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--write-every", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/v07-n2-catalano-membrane-transport-comparison.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/v07-n2-catalano-membrane-transport-comparison.csv"
        ),
    )
    args = parser.parse_args()

    if not args.feedback_exponents:
        parser.error("--feedback-exponents must not be empty")
    if any(value < 0.0 for value in args.feedback_exponents):
        parser.error("--feedback-exponents must be non-negative")
    if args.water_partial_molar_volume_cm3_mol <= 0.0:
        parser.error("--water-partial-molar-volume-cm3-mol must be positive")
    if args.activation_energy_j_mol < 0.0:
        parser.error("--activation-energy-j-mol must be non-negative")
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

    water_volume_m3_mol = args.water_partial_molar_volume_cm3_mol * 1.0e-6
    dry_reference_si = barrer_to_si_permeability(
        DRY_REFERENCE_PERMEABILITY_BARRER
    )
    linear_model, _ = make_calibrated_permeance_model(
        dry_reference_si=dry_reference_si,
        stack_temperature_k=p.stack_temperature,
        gas_constant_j_mol_k=p.gas_constant,
        activation_energy_j_mol=args.activation_energy_j_mol,
        cathode_relative_humidity=p.relative_humidity,
        membrane_thickness_m=p.membrane_thickness,
        membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        membrane_dry_density_kg_m3=p.membrane_dry_density,
        water_partial_molar_volume_m3_mol=water_volume_m3_mol,
    )
    transport_model = make_v06_transport_permeance_model(
        dry_reference_si=dry_reference_si,
        stack_temperature_k=p.stack_temperature,
        gas_constant_j_mol_k=p.gas_constant,
        faraday_c_mol=p.faraday,
        fixed_charge_mol_m3=fixed_charge,
        activation_energy_j_mol=args.activation_energy_j_mol,
        cathode_relative_humidity=p.relative_humidity,
        membrane_thickness_m=p.membrane_thickness,
        membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        membrane_dry_density_kg_m3=p.membrane_dry_density,
        membrane_water_diffusivity_m2_s=p.membrane_water_diffusivity,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
        water_partial_molar_volume_m3_mol=water_volume_m3_mol,
    )

    common_kwargs: dict[str, Any] = {
        "initial_rh": args.initial_rh,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "write_every": args.write_every,
        "volume_m3": p.anode_gas_volume_m3,
        "temperature_k": p.stack_temperature,
        "gas_constant_j_mol_k": p.gas_constant,
        "faraday_c_mol": p.faraday,
        "active_area_m2": scaling.inferred_active_area_m2,
        "fixed_charge_mol_m3": fixed_charge,
        "target_total_pressure_pa": p.anode_target_total_pressure_pa,
        "ambient_pressure_pa": p.pressure,
        "purge_interval_as": p.tech.purge_interval_as,
        "purge_duration_s": p.tech.purge_duration_max_s,
        "purge_reference_flow_slpm": p.tech.purge_rate_min_slpm_per_cell,
        "current_scale_factor": scaling.area_scale_factor,
        "n2_crossover_flux_mol_m2_s": 0.0,
        "cathode_n2_partial_pressure_pa": cathode_n2_partial_pressure,
    }

    rows: list[dict[str, Any]] = []
    for feedback_exponent in args.feedback_exponents:
        for regime in args.regimes:
            _, linear_summary = simulate_nitrogen_regime(
                regime,
                closure[regime],
                hydrogen_feedback_exponent=feedback_exponent,
                n2_permeance_model=linear_model,
                **common_kwargs,
            )
            _, transport_summary = simulate_nitrogen_regime(
                regime,
                closure[regime],
                hydrogen_feedback_exponent=feedback_exponent,
                n2_state_permeance_model=transport_model,
                **common_kwargs,
            )

            linear_flux = float(
                linear_summary["mean_n2_crossover_flux_mol_m2_s"]
            )
            transport_flux = float(
                transport_summary["mean_n2_crossover_flux_mol_m2_s"]
            )
            linear_xn2 = float(linear_summary["max_nitrogen_mole_fraction"])
            transport_xn2 = float(
                transport_summary["max_nitrogen_mole_fraction"]
            )

            row = {
                "regime": regime,
                "hydrogen_feedback_exponent": feedback_exponent,
                "water_partial_molar_volume_cm3_mol": (
                    args.water_partial_molar_volume_cm3_mol
                ),
                "activation_energy_j_mol": args.activation_energy_j_mol,
                "linear_jmean_mol_m2_s": linear_flux,
                "v06_transport_jmean_mol_m2_s": transport_flux,
                "jmean_ratio_transport_to_linear": (
                    transport_flux / linear_flux
                ),
                "jmean_change_fraction": relative_change(
                    transport_flux,
                    linear_flux,
                ),
                "linear_xn2_max": linear_xn2,
                "v06_transport_xn2_max": transport_xn2,
                "xn2_ratio_transport_to_linear": (
                    transport_xn2 / linear_xn2
                ),
                "xn2_change_fraction": relative_change(
                    transport_xn2,
                    linear_xn2,
                ),
                "linear_purge_count": linear_summary["purge_count"],
                "v06_transport_purge_count": transport_summary["purge_count"],
                "linear_mean_purge_period_s": (
                    linear_summary["mean_purge_period_s"]
                ),
                "v06_transport_mean_purge_period_s": (
                    transport_summary["mean_purge_period_s"]
                ),
                "linear_final_rh": linear_summary["final_relative_humidity"],
                "v06_transport_final_rh": (
                    transport_summary["final_relative_humidity"]
                ),
                "linear_h2_balance_error_mol": (
                    linear_summary["h2_balance_error_mol"]
                ),
                "v06_transport_h2_balance_error_mol": (
                    transport_summary["h2_balance_error_mol"]
                ),
                "linear_n2_balance_error_mol": (
                    linear_summary["n2_balance_error_mol"]
                ),
                "v06_transport_n2_balance_error_mol": (
                    transport_summary["n2_balance_error_mol"]
                ),
                "linear_water_balance_error_mol": (
                    linear_summary["water_balance_error_mol"]
                ),
                "v06_transport_water_balance_error_mol": (
                    transport_summary["water_balance_error_mol"]
                ),
            }
            rows.append(row)
            print(
                f"{regime} gamma={feedback_exponent:.1f}: "
                f"J {linear_flux:.3e}->{transport_flux:.3e} "
                f"({100.0 * row['jmean_change_fraction']:+.1f}%), "
                f"xN2,max {linear_xn2:.5f}->{transport_xn2:.5f} "
                f"({100.0 * row['xn2_change_fraction']:+.1f}%)",
                flush=True,
            )

    output = {
        "schema_version": 1,
        "model": "v07-n2-catalano-membrane-transport-comparison",
        "comparison": "linear_profile_vs_v06_finite_transfer_profile",
        "closure_source": str(args.closure_csv),
        "water_partial_molar_volume_cm3_mol": (
            args.water_partial_molar_volume_cm3_mol
        ),
        "activation_energy_j_mol": args.activation_energy_j_mol,
        "dry_reference_permeability_barrer": (
            DRY_REFERENCE_PERMEABILITY_BARRER
        ),
        "anchor_water_activity": ANCHOR_WATER_ACTIVITY,
        "anchor_permeability_factor": ANCHOR_PERMEABILITY_FACTOR,
        "maximum_permeability_factor": MAXIMUM_PERMEABILITY_FACTOR,
        "membrane_profile_points": MEMBRANE_PROFILE_POINTS,
        "membrane_water_diffusivity_m2_s": p.membrane_water_diffusivity,
        "anode_transfer_coefficient_m_s": (
            p.anode_water_transfer_coefficient
        ),
        "feedback_exponents": args.feedback_exponents,
        "regimes": args.regimes,
        "stop_time_s": args.stop_time,
        "dt_s": args.dt,
        "summaries": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
