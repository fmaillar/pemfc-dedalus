"""Bracket a common cell voltage for the V1.0 imposed-current closure."""

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
    """Return the 3-point Simpson mean over xi in [0, 1]."""
    if len(values) != 3:
        raise ValueError("Simpson mean requires exactly three values")
    return (values[0] + 4.0 * values[1] + values[2]) / 6.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-a", type=float, default=26.04)
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--ntu", type=float, default=3.0)
    parser.add_argument(
        "--voltages-v",
        nargs="+",
        type=float,
        default=[0.74, 0.77, 0.80],
    )
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--stop-time", type=float, default=1.0e-3)
    parser.add_argument("--max-dt", type=float, default=2.0e-6)
    parser.add_argument("--profile-points", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output-v10-voltage-bracket"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v10-voltage-bracket.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v10-voltage-bracket.csv"),
    )
    args = parser.parse_args()

    base = CathodeParameters()
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
        thermal_params.membrane_current_density
        * args.current_a
        / thermal_params.stack_current_a
    )

    rows: list[dict[str, Any]] = []
    voltage_summaries: list[dict[str, float]] = []

    for voltage_v in args.voltages_v:
        params = replace(
            thermal_params,
            cathode_solid_potential=voltage_v,
        )
        local_current_densities: list[float] = []

        for xi in xis:
            run_dir = (
                args.work_dir
                / f"voltage-{voltage_v:.4f}"
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
                    "voltage_v": voltage_v,
                    "slice_xi": xi,
                    "oxygen_feed_concentration_mol_m3": local_feed[xi],
                    "model_current_density_a_m2": current_density,
                    "target_current_density_a_m2": target_current_density,
                    "mean_eta_v": _last_scalar(run_dir, "mean_eta"),
                }
            )

        mean_current_density = simpson_mean(local_current_densities)
        voltage_summaries.append(
            {
                "voltage_v": voltage_v,
                "mean_current_density_a_m2": mean_current_density,
                "relative_error": (
                    mean_current_density - target_current_density
                )
                / target_current_density,
            }
        )

    ordered = sorted(voltage_summaries, key=lambda item: item["voltage_v"])
    monotonic_decreasing = all(
        right["mean_current_density_a_m2"]
        < left["mean_current_density_a_m2"]
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    bracketed = (
        min(item["mean_current_density_a_m2"] for item in ordered)
        <= target_current_density
        <= max(item["mean_current_density_a_m2"] for item in ordered)
    )

    output = {
        "schema_version": 1,
        "model": "v10-common-voltage-bracket",
        "current_a": args.current_a,
        "target_current_density_a_m2": target_current_density,
        "bol_voltage_v": bol_voltage_v,
        "monotonic_decreasing": monotonic_decreasing,
        "target_bracketed": bracketed,
        "voltage_summaries": voltage_summaries,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"monotonic_decreasing={monotonic_decreasing} "
        f"target_bracketed={bracketed}",
        flush=True,
    )
    for item in ordered:
        print(
            f"V={item['voltage_v']:.4f} V "
            f"j_bar={item['mean_current_density_a_m2']:.2f} A/m2 "
            f"error={100.0 * item['relative_error']:+.2f}%",
            flush=True,
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
