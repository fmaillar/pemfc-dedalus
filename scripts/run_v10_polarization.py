"""Build a thermo-coupled galvanostatic polarization curve for V1.0."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pemfc_dedalus.cathode_airflow import ideal_gas_species_concentration_mol_m3
from pemfc_dedalus.cathode_airflow_1d import solve_streamwise_finite_thermal_profile
from pemfc_dedalus.cathode_electrochem_3d import run as run_cathode_electrochem
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import open_cathode_airflow_target


def _last_scalar(path: Path, task: str) -> float:
    files = sorted((path / "scalars").glob("scalars_s*.h5"))
    if not files:
        raise FileNotFoundError(f"no Dedalus scalar files found in {path / 'scalars'}")
    with h5py.File(files[-1], "r") as handle:
        values = np.asarray(handle[f"tasks/{task}"])
        return float(np.ravel(values[-1])[0])


def simpson_mean(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("Simpson mean requires exactly three values")
    return (values[0] + 4.0 * values[1] + values[2]) / 6.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--currents-a",
        nargs="+",
        type=float,
        default=[7.3, 14.5, 26.04],
    )
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--ntu", type=float, default=3.0)
    parser.add_argument("--bracket-step-v", type=float, default=0.03)
    parser.add_argument("--max-bracket-steps", type=int, default=6)
    parser.add_argument("--relative-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--stop-time", type=float, default=1.0e-3)
    parser.add_argument("--max-dt", type=float, default=2.0e-6)
    parser.add_argument("--profile-points", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output-v10-polarization"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v10-polarization.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v10-polarization.csv"),
    )
    args = parser.parse_args()

    if args.bracket_step_v <= 0.0:
        raise ValueError("bracket-step-v must be positive")
    if args.relative_tolerance <= 0.0:
        raise ValueError("relative-tolerance must be positive")

    base = CathodeParameters()
    reference_current_a = base.stack_current_a
    area_m2 = base.length_x * base.length_y
    xis = [0.0, 0.5, 1.0]

    rows: list[dict[str, Any]] = []
    solutions: list[dict[str, Any]] = []

    for current_a in args.currents_a:
        target_temperature_c = base.tech.optimum_stack_temperature_c(current_a)
        target_temperature_k = 273.15 + target_temperature_c
        inlet_temperature_k = 273.15 + args.inlet_temperature_c
        bol_voltage_v = base.tech.bol_typical_cell_voltage_v(current_a)
        target_current_density = (
            base.membrane_current_density
            * current_a
            / reference_current_a
        )
        operating_base = replace(
            base,
            stack_current_a=current_a_bound,
            stack_temperature=target_temperature_k,
            cathode_solid_potential=bol_voltage_v,
        )

        evaluation_index = 0

        def evaluate(
            voltage_v: float,
            label: str,
            *,
            current_a_bound: float = current_a,
            operating_base_bound: CathodeParameters = operating_base,
            target_temperature_k_bound: float = target_temperature_k,
            inlet_temperature_k_bound: float = inlet_temperature_k,
            target_current_density_bound: float = target_current_density,
        ) -> float:
            nonlocal evaluation_index
            evaluation_index += 1

            heat_rejection_w = operating_base_bound.stack.heat_rejection_w(
                current_a_bound,
                voltage_v,
            )
            airflow = open_cathode_airflow_target(
                heat_rejection_w=heat_rejection_w,
                inlet_temperature_k=inlet_temperature_k_bound,
                target_stack_temperature_k=target_temperature_k_bound,
                stoichiometric_floor_slpm=operating_base_bound.stack.coolant_air_target_slpm(
                    current_a_bound
                ),
            )
            if airflow.target_air_flow_slpm is None:
                raise RuntimeError(
                    f"thermally unreachable current={current_a_bound} A "
                    f"voltage={voltage_v} V"
                )

            profile = solve_streamwise_finite_thermal_profile(
                current_a=current_a_bound,
                total_air_flow_slpm=airflow.target_air_flow_slpm,
                n_cells=operating_base_bound.stack.n_cells,
                oxygen_mole_fraction=operating_base_bound.oxygen_mole_fraction,
                faraday_c_mol=operating_base_bound.faraday,
                inlet_temperature_k=inlet_temperature_k_bound,
                stack_temperature_k=target_temperature_k,
                ntu=args.ntu,
                points=args.profile_points,
            )
            concentration = ideal_gas_species_concentration_mol_m3(
                mole_fraction=profile.oxygen_mole_fraction,
                pressure_pa=operating_base_bound.pressure,
                temperature_k=profile.air_temperature_k,
                gas_constant_j_mol_k=operating_base_bound.gas_constant,
            )
            local_feed = {
                xi: float(
                    np.interp(
                        xi,
                        profile.streamwise_fraction,
                        concentration,
                    )
                )
                for xi in xis
            }

            params = replace(
                operating_base_bound,
                cathode_solid_potential=voltage_v,
            )
            local_current_densities: list[float] = []

            for xi in xis:
                run_dir = (
                    args.work_dir
                    / f"current-{current_a_bound:.2f}"
                    / f"eval-{evaluation_index:02d}-{label}"
                    / f"xi-{xi:.1f}"
                )
                if run_dir.exists():
                    shutil.rmtree(run_dir)

                run_cathode_electrochem(
                    params=params,
                    nx=args.nx,
                    ny=args.ny,
                    nz=args.nz,
                    stop_time=args.stop_time,
                    max_dt=args.max_dt,
                    output_dir=run_dir,
                    scalar_dt=max(args.stop_time / 20.0, 1.0e-7),
                    oxygen_feed_concentration=local_feed[xi],
                )
                reaction_current = _last_scalar(run_dir, "total_reaction_current")
                current_density = reaction_current / area_m2
                local_current_densities.append(current_density)
                rows.append(
                    {
                        "current_a": current_a_bound,
                        "evaluation": label,
                        "voltage_v": voltage_v,
                        "slice_xi": xi,
                        "oxygen_feed_concentration_mol_m3": local_feed[xi],
                        "air_flow_slpm": airflow.target_air_flow_slpm,
                        "heat_rejection_w": heat_rejection_w,
                        "model_current_density_a_m2": current_density,
                        "target_current_density_a_m2": target_current_density_bound,
                        "mean_eta_v": _last_scalar(run_dir, "mean_eta"),
                    }
                )

            return simpson_mean(local_current_densities)

        bol_j = evaluate(bol_voltage_v, "bol")
        bol_error = (bol_j - target_current_density) / target_current_density

        if abs(bol_error) <= args.relative_tolerance:
            low_v = bol_voltage_v
            high_v = bol_voltage_v
            solution_v = bol_voltage_v
            solution_j = bol_j
            history: list[dict[str, float | int]] = []
            converged = True
        else:
            direction = 1.0 if bol_j > target_current_density else -1.0
            trial_v = bol_voltage_v
            trial_j = bol_j
            bracket_found = False

            for step in range(1, args.max_bracket_steps + 1):
                trial_v = bol_voltage_v + direction * args.bracket_step_v * step
                if not 0.0 < trial_v < operating_base.equilibrium_potential:
                    raise RuntimeError("adaptive voltage bracket left physical range")
                trial_j = evaluate(trial_v, f"bracket-{step:02d}")

                if (
                    bol_j >= target_current_density >= trial_j
                    or trial_j >= target_current_density >= bol_j
                ):
                    bracket_found = True
                    break

            if not bracket_found:
                raise RuntimeError(
                    f"could not bracket target at current={current_a} A"
                )

            low_v = min(bol_voltage_v, trial_v)
            high_v = max(bol_voltage_v, trial_v)
            low_j = bol_j if bol_voltage_v == low_v else trial_j
            high_j = bol_j if bol_voltage_v == high_v else trial_j

            if low_j < high_j:
                low_v, high_v = high_v, low_v
                low_j, high_j = high_j, low_j

            history = []
            converged = False
            solution_v = bol_voltage_v
            solution_j = bol_j

            for iteration in range(1, args.max_iterations + 1):
                mid_v = 0.5 * (low_v + high_v)
                mid_j = evaluate(mid_v, f"iteration-{iteration:02d}")
                relative_error = (
                    mid_j - target_current_density
                ) / target_current_density

                history.append(
                    {
                        "iteration": iteration,
                        "voltage_v": mid_v,
                        "mean_current_density_a_m2": mid_j,
                        "relative_error": relative_error,
                    }
                )

                solution_v = mid_v
                solution_j = mid_j

                if abs(relative_error) <= args.relative_tolerance:
                    converged = True
                    break

                if mid_j > target_current_density:
                    low_v = mid_v
                    low_j = mid_j
                else:
                    high_v = mid_v
                    high_j = mid_j

        final_error = (
            solution_j - target_current_density
        ) / target_current_density

        solutions.append(
            {
                "current_a": current_a_bound,
                "target_temperature_c": target_temperature_c,
                "target_current_density_a_m2": target_current_density_bound,
                "bol_voltage_v": bol_voltage_v,
                "bol_mean_current_density_a_m2": bol_j,
                "bol_relative_error": bol_error,
                "converged": converged,
                "solution_voltage_v": solution_v,
                "solution_mean_current_density_a_m2": solution_j,
                "solution_relative_error": final_error,
                "voltage_shift_from_bol_v": solution_v - bol_voltage_v,
                "iteration_count": len(history),
                "history": history,
            }
        )

    ordered = sorted(solutions, key=lambda item: float(item["current_a"]))
    voltage_decreases_with_current = all(
        float(right["solution_voltage_v"]) < float(left["solution_voltage_v"])
        for left, right in zip(ordered, ordered[1:], strict=False)
    )

    output = {
        "schema_version": 1,
        "model": "v10-thermo-coupled-galvanostatic-polarization",
        "inlet_temperature_c": args.inlet_temperature_c,
        "ntu": args.ntu,
        "all_converged": all(bool(item["converged"]) for item in ordered),
        "voltage_decreases_with_current": voltage_decreases_with_current,
        "solutions": solutions,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for item in ordered:
        print(
            f"I={item['current_a']:.2f} A "
            f"V={item['solution_voltage_v']:.6f} V "
            f"error={100.0 * item['solution_relative_error']:+.3f}% "
            f"dV_BOL={1000.0 * item['voltage_shift_from_bol_v']:+.2f} mV",
            flush=True,
        )
    print(
        f"all_converged={output['all_converged']} "
        f"voltage_decreases_with_current={voltage_decreases_with_current}",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
