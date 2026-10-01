"""Solve common cell voltage by bisection for imposed current."""

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
from pemfc_dedalus.cathode_airflow_1d import (
    solve_streamwise_finite_thermal_profile,
)
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
    parser.add_argument("--current-a", type=float, default=26.04)
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--ntu", type=float, default=3.0)
    parser.add_argument("--voltage-low-v", type=float, default=0.74)
    parser.add_argument("--voltage-high-v", type=float, default=0.77)
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
        default=Path("output-v10-voltage-bisection"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v10-voltage-bisection.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v10-voltage-bisection.csv"),
    )
    args = parser.parse_args()

    if args.voltage_low_v >= args.voltage_high_v:
        raise ValueError("voltage-low-v must be smaller than voltage-high-v")
    if args.relative_tolerance <= 0.0:
        raise ValueError("relative-tolerance must be positive")

    base = CathodeParameters()
    reference_current_a = base.stack_current_a
    target_temperature_c = base.tech.optimum_stack_temperature_c(args.current_a)
    target_temperature_k = 273.15 + target_temperature_c
    inlet_temperature_k = 273.15 + args.inlet_temperature_c
    bol_voltage_v = base.tech.bol_typical_cell_voltage_v(args.current_a)

    thermal_params = replace(
        base,
        stack_current_a=args.current_a,
        stack_temperature=target_temperature_k,
        cathode_solid_potential=bol_voltage_v,
    )
    heat_rejection_w = thermal_params.stack.heat_rejection_w(
        args.current_a,
        bol_voltage_v,
    )
    airflow = open_cathode_airflow_target(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        target_stack_temperature_k=target_temperature_k,
        stoichiometric_floor_slpm=thermal_params.stack.coolant_air_target_slpm(
            args.current_a
        ),
    )
    if airflow.target_air_flow_slpm is None:
        raise RuntimeError("selected operating point is thermally unreachable")

    profile = solve_streamwise_finite_thermal_profile(
        current_a=args.current_a,
        total_air_flow_slpm=airflow.target_air_flow_slpm,
        n_cells=thermal_params.stack.n_cells,
        oxygen_mole_fraction=thermal_params.oxygen_mole_fraction,
        faraday_c_mol=thermal_params.faraday,
        inlet_temperature_k=inlet_temperature_k,
        stack_temperature_k=target_temperature_k,
        ntu=args.ntu,
        points=args.profile_points,
    )
    concentration = ideal_gas_species_concentration_mol_m3(
        mole_fraction=profile.oxygen_mole_fraction,
        pressure_pa=thermal_params.pressure,
        temperature_k=profile.air_temperature_k,
        gas_constant_j_mol_k=thermal_params.gas_constant,
    )

    xis = [0.0, 0.5, 1.0]
    local_feed = {
        xi: float(np.interp(xi, profile.streamwise_fraction, concentration))
        for xi in xis
    }

    area_m2 = thermal_params.length_x * thermal_params.length_y
    target_current_density = (
        base.membrane_current_density
        * args.current_a
        / reference_current_a
    )

    rows: list[dict[str, Any]] = []

    def evaluate(voltage_v: float, label: str) -> float:
        params = replace(
            thermal_params,
            cathode_solid_potential=voltage_v,
        )
        local_current_densities: list[float] = []

        for xi in xis:
            run_dir = (
                args.work_dir
                / label
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
                    "evaluation": label,
                    "voltage_v": voltage_v,
                    "slice_xi": xi,
                    "oxygen_feed_concentration_mol_m3": local_feed[xi],
                    "model_current_density_a_m2": current_density,
                    "target_current_density_a_m2": target_current_density,
                    "mean_eta_v": _last_scalar(run_dir, "mean_eta"),
                }
            )

        return simpson_mean(local_current_densities)

    low_v = args.voltage_low_v
    high_v = args.voltage_high_v
    low_j = evaluate(low_v, "bracket-low")
    high_j = evaluate(high_v, "bracket-high")

    if not low_j >= target_current_density >= high_j:
        raise RuntimeError(
            "initial voltage bracket does not contain target current density"
        )

    history: list[dict[str, float | int]] = []
    converged = False
    solution_v = high_v
    solution_j = high_j

    for iteration in range(1, args.max_iterations + 1):
        mid_v = 0.5 * (low_v + high_v)
        mid_j = evaluate(mid_v, f"iteration-{iteration:02d}")
        relative_error = (mid_j - target_current_density) / target_current_density

        history.append(
            {
                "iteration": iteration,
                "voltage_v": mid_v,
                "mean_current_density_a_m2": mid_j,
                "relative_error": relative_error,
                "bracket_low_v": low_v,
                "bracket_high_v": high_v,
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

    final_relative_error = (
        solution_j - target_current_density
    ) / target_current_density

    output = {
        "schema_version": 1,
        "model": "v10-common-voltage-bisection",
        "current_a": args.current_a,
        "target_current_density_a_m2": target_current_density,
        "bol_voltage_v": bol_voltage_v,
        "converged": converged,
        "relative_tolerance": args.relative_tolerance,
        "iteration_count": len(history),
        "solution_voltage_v": solution_v,
        "solution_mean_current_density_a_m2": solution_j,
        "solution_relative_error": final_relative_error,
        "voltage_shift_from_bol_v": solution_v - bol_voltage_v,
        "history": history,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"converged={converged} iterations={len(history)} "
        f"V={solution_v:.6f} V "
        f"j_bar={solution_j:.2f} A/m2 "
        f"error={100.0 * final_relative_error:+.3f}%",
        flush=True,
    )
    print(
        f"V_BOL={bol_voltage_v:.6f} V "
        f"delta_V={1000.0 * (solution_v - bol_voltage_v):+.2f} mV",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
