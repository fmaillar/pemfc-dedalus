"""Sensitivity of Catalano N2 crossover to V0.6 membrane-water parameters.

This campaign varies the membrane water diffusivity and the finite anode
transfer coefficient used by the V0.6 steady membrane profile while keeping
the Catalano N2 permeability calibration fixed.

Each (regime, gamma, D_water, k_a) case is independent and can be run in
parallel.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.anode_nitrogen import (
    humid_air_nitrogen_partial_pressure_pa,
)
from pemfc_dedalus.gas_permeability import barrer_to_si_permeability
from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.scaling import infer_active_area_scaling
from scripts.run_v07_anode_h2 import (
    REGIMES,
    interpolate_flux_and_current,
    load_h2_closure,
)
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime
from scripts.run_v07_n2_catalano_calibrated import (
    DRY_REFERENCE_PERMEABILITY_BARRER,
)
from scripts.run_v07_n2_catalano_membrane_transport import (
    DEFAULT_ACTIVATION_ENERGY_J_MOL,
    DEFAULT_WATER_PARTIAL_MOLAR_VOLUME_CM3_MOL,
    make_v06_transport_permeance_model,
)

DEFAULT_DIFFUSIVITIES_M2_S = [1.0e-10, 2.0e-10, 4.0e-10]
DEFAULT_TRANSFER_COEFFICIENTS_M_S = [1.0e-6, 2.0e-6, 4.0e-6]
DEFAULT_FEEDBACK_EXPONENTS = [0.0, 1.0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty membrane-transport sensitivity")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_sensitivity_case(task: dict[str, Any]) -> dict[str, Any]:
    """Run one independent V0.6 membrane-transport sensitivity case."""
    p = CathodeParameters()
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    dry_reference_si = barrer_to_si_permeability(
        DRY_REFERENCE_PERMEABILITY_BARRER
    )
    water_volume_m3_mol = (
        float(task["water_partial_molar_volume_cm3_mol"]) * 1.0e-6
    )
    diffusivity = float(task["diffusivity_m2_s"])
    transfer_coefficient = float(task["transfer_coefficient_m_s"])

    transport_model = make_v06_transport_permeance_model(
        dry_reference_si=dry_reference_si,
        stack_temperature_k=p.stack_temperature,
        gas_constant_j_mol_k=p.gas_constant,
        faraday_c_mol=p.faraday,
        fixed_charge_mol_m3=fixed_charge,
        activation_energy_j_mol=float(task["activation_energy_j_mol"]),
        cathode_relative_humidity=p.relative_humidity,
        membrane_thickness_m=p.membrane_thickness,
        membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        membrane_dry_density_kg_m3=p.membrane_dry_density,
        membrane_water_diffusivity_m2_s=diffusivity,
        anode_transfer_coefficient_m_s=transfer_coefficient,
        water_partial_molar_volume_m3_mol=water_volume_m3_mol,
    )

    regime = str(task["regime"])
    feedback_exponent = float(task["feedback_exponent"])
    _, summary = simulate_nitrogen_regime(
        regime,
        task["closure"],
        hydrogen_feedback_exponent=feedback_exponent,
        n2_state_permeance_model=transport_model,
        **task["common_kwargs"],
    )

    return {
        "regime": regime,
        "hydrogen_feedback_exponent": feedback_exponent,
        "membrane_water_diffusivity_m2_s": diffusivity,
        "anode_transfer_coefficient_m_s": transfer_coefficient,
        "mean_n2_crossover_flux_mol_m2_s": (
            summary["mean_n2_crossover_flux_mol_m2_s"]
        ),
        "max_nitrogen_mole_fraction": summary["max_nitrogen_mole_fraction"],
        "min_hydrogen_mole_fraction": summary["min_hydrogen_mole_fraction"],
        "mean_cell_current_a": summary["mean_cell_current_a"],
        "min_hydrogen_feedback_factor": (
            summary["min_hydrogen_feedback_factor"]
        ),
        "purge_count": summary["purge_count"],
        "mean_purge_period_s": summary["mean_purge_period_s"],
        "final_relative_humidity": summary["final_relative_humidity"],
        "h2_balance_error_mol": summary["h2_balance_error_mol"],
        "n2_balance_error_mol": summary["n2_balance_error_mol"],
        "water_balance_error_mol": summary["water_balance_error_mol"],
    }


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
        "--diffusivities-m2-s",
        nargs="+",
        type=float,
        default=DEFAULT_DIFFUSIVITIES_M2_S,
    )
    parser.add_argument(
        "--transfer-coefficients-m-s",
        nargs="+",
        type=float,
        default=DEFAULT_TRANSFER_COEFFICIENTS_M_S,
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
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/v07-n2-catalano-membrane-transport-sensitivity.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/v07-n2-catalano-membrane-transport-sensitivity.csv"
        ),
    )
    args = parser.parse_args()

    if not args.diffusivities_m2_s or any(
        value <= 0.0 for value in args.diffusivities_m2_s
    ):
        parser.error("--diffusivities-m2-s must contain positive values")
    if not args.transfer_coefficients_m_s or any(
        value < 0.0 for value in args.transfer_coefficients_m_s
    ):
        parser.error(
            "--transfer-coefficients-m-s must contain non-negative values"
        )
    if not args.feedback_exponents or any(
        value < 0.0 for value in args.feedback_exponents
    ):
        parser.error("--feedback-exponents must contain non-negative values")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
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

    tasks: list[dict[str, Any]] = []
    for feedback_exponent in args.feedback_exponents:
        for regime in args.regimes:
            for diffusivity in args.diffusivities_m2_s:
                for transfer_coefficient in args.transfer_coefficients_m_s:
                    tasks.append(
                        {
                            "regime": regime,
                            "feedback_exponent": feedback_exponent,
                            "diffusivity_m2_s": diffusivity,
                            "transfer_coefficient_m_s": transfer_coefficient,
                            "closure": closure[regime],
                            "common_kwargs": common_kwargs,
                            "water_partial_molar_volume_cm3_mol": (
                                args.water_partial_molar_volume_cm3_mol
                            ),
                            "activation_energy_j_mol": (
                                args.activation_energy_j_mol
                            ),
                        }
                    )

    worker_count = min(args.jobs, len(tasks))
    if worker_count == 1:
        rows = [run_sensitivity_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(run_sensitivity_case, tasks))

    for row in rows:
        print(
            f"{row['regime']} gamma={row['hydrogen_feedback_exponent']:.1f} "
            f"D={row['membrane_water_diffusivity_m2_s']:.2e} "
            f"ka={row['anode_transfer_coefficient_m_s']:.2e}: "
            f"Jmean={row['mean_n2_crossover_flux_mol_m2_s']:.3e} "
            f"xN2,max={row['max_nitrogen_mole_fraction']:.5f}",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "model": "v07-n2-catalano-membrane-transport-sensitivity",
        "closure_source": str(args.closure_csv),
        "water_partial_molar_volume_cm3_mol": (
            args.water_partial_molar_volume_cm3_mol
        ),
        "activation_energy_j_mol": args.activation_energy_j_mol,
        "diffusivities_m2_s": args.diffusivities_m2_s,
        "transfer_coefficients_m_s": args.transfer_coefficients_m_s,
        "feedback_exponents": args.feedback_exponents,
        "regimes": args.regimes,
        "jobs": worker_count,
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
