"""Diagnose V0.7 membrane water profiles and N2 transport resistance.

This post-processing script reads the completed D_water/k_a sensitivity table
and reconstructs the representative V0.6 membrane profile for each row using
its final anode RH and mean cell current. It reports dimensionless transport
groups and how the local N2 resistance is distributed through the membrane.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

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
from scripts.run_v07_anode_h2 import interpolate_flux_and_current, load_h2_closure
from scripts.run_v07_n2_catalano_calibrated import (
    ANCHOR_PERMEABILITY_FACTOR,
    ANCHOR_WATER_ACTIVITY,
    DRY_REFERENCE_PERMEABILITY_BARRER,
    MAXIMUM_PERMEABILITY_FACTOR,
    MEMBRANE_PROFILE_POINTS,
    REFERENCE_TEMPERATURE_K,
)
from scripts.run_v07_n2_catalano_membrane_transport import (
    DEFAULT_ACTIVATION_ENERGY_J_MOL,
    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
)


def scalar_water_content(activity: float) -> float:
    """Return scalar Nafion water content for scalar water activity."""
    return float(membrane_water_content_from_activity(activity))


def resistance_fraction(
    z_m: np.ndarray,
    permeability_profile: np.ndarray,
    *,
    start_fraction: float,
    end_fraction: float,
) -> float:
    """Return the share of total through-plane resistance in one interval."""
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("fractions must satisfy 0 <= start < end <= 1")
    z = np.asarray(z_m, dtype=float)
    permeability = np.asarray(permeability_profile, dtype=float)
    if z.shape != permeability.shape:
        raise ValueError("z_m and permeability_profile must have same shape")
    if np.any(permeability <= 0.0):
        raise ValueError("permeability_profile must be positive")

    resistance_density = 1.0 / permeability
    total_resistance = float(np.trapezoid(resistance_density, z))

    length = float(z[-1] - z[0])
    lower = z[0] + start_fraction * length
    upper = z[0] + end_fraction * length
    mask = (z >= lower) & (z <= upper)
    z_segment = z[mask]
    r_segment = resistance_density[mask]

    if z_segment.size < 2:
        raise ValueError("profile grid too coarse for requested interval")

    segment_resistance = float(np.trapezoid(r_segment, z_segment))
    return segment_resistance / total_resistance


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty membrane-profile diagnostics")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "results/v07-n2-catalano-membrane-transport-sensitivity.csv"
        ),
    )
    parser.add_argument(
        "--closure-csv",
        type=Path,
        default=Path("results/v06-rha-sensitivity.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/v07-n2-catalano-membrane-profile-diagnostics.csv"
        ),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    closure = load_h2_closure(args.closure_csv)
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

    water_volume_m3_mol = (
        DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL * 1.0e-6
    )
    lambda_dry = scalar_water_content(0.0)
    lambda_anchor = scalar_water_content(ANCHOR_WATER_ACTIVITY)
    lambda_cathode = scalar_water_content(p.relative_humidity)
    phi_dry = membrane_water_volume_fraction_from_partial_molar_volume(
        lambda_dry,
        membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        membrane_dry_density_kg_m3=p.membrane_dry_density,
        water_partial_molar_volume_m3_mol=water_volume_m3_mol,
    )
    phi_anchor = membrane_water_volume_fraction_from_partial_molar_volume(
        lambda_anchor,
        membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        membrane_dry_density_kg_m3=p.membrane_dry_density,
        water_partial_molar_volume_m3_mol=water_volume_m3_mol,
    )
    dry_at_temperature = arrhenius_permeability(
        barrer_to_si_permeability(DRY_REFERENCE_PERMEABILITY_BARRER),
        p.stack_temperature,
        REFERENCE_TEMPERATURE_K,
        DEFAULT_ACTIVATION_ENERGY_J_MOL,
        p.gas_constant,
    )
    z_membrane = np.linspace(
        0.0,
        p.membrane_thickness,
        MEMBRANE_PROFILE_POINTS,
    )

    input_rows = list(csv.DictReader(args.input_csv.open(newline="")))
    rows: list[dict[str, Any]] = []

    for source in input_rows:
        diffusivity = float(source["membrane_water_diffusivity_m2_s"])
        transfer_coefficient = float(
            source["anode_transfer_coefficient_m_s"]
        )
        final_rh = float(source["final_relative_humidity"])
        mean_cell_current = float(source["mean_cell_current_a"])
        current_density = (
            mean_cell_current / scaling.inferred_active_area_m2
        )
        drag_velocity = electro_osmotic_lambda_velocity(
            current_density,
            p.faraday,
            fixed_charge,
        )
        peclet = (
            drag_velocity * p.membrane_thickness / diffusivity
        )
        transfer_number = (
            transfer_coefficient * p.membrane_thickness / diffusivity
        )
        lambda_anode_equilibrium = scalar_water_content(final_rh)
        lambda_profile = steady_membrane_water_profile_anode_transfer(
            z_membrane,
            lambda_cathode=lambda_cathode,
            lambda_anode_equilibrium=lambda_anode_equilibrium,
            diffusivity_m2_s=diffusivity,
            drag_velocity_m_s=drag_velocity,
            anode_transfer_coefficient_m_s=transfer_coefficient,
        )

        permeability_values: list[float] = []
        phi_values: list[float] = []
        for lambda_local in lambda_profile:
            phi_water = (
                membrane_water_volume_fraction_from_partial_molar_volume(
                    float(lambda_local),
                    membrane_equivalent_weight_kg_mol=(
                        p.membrane_equivalent_weight
                    ),
                    membrane_dry_density_kg_m3=p.membrane_dry_density,
                    water_partial_molar_volume_m3_mol=water_volume_m3_mol,
                )
            )
            humidity_factor = calibrated_exponential_permeability_multiplier(
                phi_water,
                baseline_fraction=phi_dry,
                anchor_fraction=phi_anchor,
                anchor_factor=ANCHOR_PERMEABILITY_FACTOR,
                maximum_factor=MAXIMUM_PERMEABILITY_FACTOR,
            )
            phi_values.append(phi_water)
            permeability_values.append(
                dry_at_temperature * humidity_factor
            )

        permeability_profile = np.asarray(permeability_values)
        phi_profile = np.asarray(phi_values)
        permeance = membrane_permeance_from_permeability_profile(
            permeability_values,
            p.membrane_thickness,
        )

        rows.append(
            {
                "regime": source["regime"],
                "hydrogen_feedback_exponent": float(
                    source["hydrogen_feedback_exponent"]
                ),
                "membrane_water_diffusivity_m2_s": diffusivity,
                "anode_transfer_coefficient_m_s": transfer_coefficient,
                "water_peclet": peclet,
                "anode_transfer_number": transfer_number,
                "lambda_anode_equilibrium": lambda_anode_equilibrium,
                "lambda_anode_profile": float(lambda_profile[0]),
                "lambda_cathode": float(lambda_profile[-1]),
                "lambda_mean": float(np.mean(lambda_profile)),
                "lambda_min": float(np.min(lambda_profile)),
                "lambda_max": float(np.max(lambda_profile)),
                "phi_water_mean": float(np.mean(phi_profile)),
                "phi_water_min": float(np.min(phi_profile)),
                "phi_water_max": float(np.max(phi_profile)),
                "permeability_min_si": float(
                    np.min(permeability_profile)
                ),
                "permeability_max_si": float(
                    np.max(permeability_profile)
                ),
                "permeance_mol_m2_s_pa": permeance,
                "anode_half_resistance_fraction": resistance_fraction(
                    z_membrane,
                    permeability_profile,
                    start_fraction=0.0,
                    end_fraction=0.5,
                ),
                "cathode_half_resistance_fraction": resistance_fraction(
                    z_membrane,
                    permeability_profile,
                    start_fraction=0.5,
                    end_fraction=1.0,
                ),
                "anode_quarter_resistance_fraction": resistance_fraction(
                    z_membrane,
                    permeability_profile,
                    start_fraction=0.0,
                    end_fraction=0.25,
                ),
                "cathode_quarter_resistance_fraction": resistance_fraction(
                    z_membrane,
                    permeability_profile,
                    start_fraction=0.75,
                    end_fraction=1.0,
                ),
                "source_mean_n2_crossover_flux_mol_m2_s": float(
                    source["mean_n2_crossover_flux_mol_m2_s"]
                ),
                "source_max_nitrogen_mole_fraction": float(
                    source["max_nitrogen_mole_fraction"]
                ),
            }
        )

    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
