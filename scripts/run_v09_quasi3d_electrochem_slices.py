"""Run quasi-3D cathode electrochemistry slices for V0.9."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-a", type=float, default=26.04)
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--ntu", type=float, default=3.0)
    parser.add_argument("--slice-xi", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--profile-points", type=int, default=64)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--stop-time", type=float, default=5.0e-4)
    parser.add_argument("--max-dt", type=float, default=2.0e-6)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output-v09-electrochem-slices"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v09-quasi3d-electrochem-slices.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v09-quasi3d-electrochem-slices.csv"),
    )
    args = parser.parse_args()

    if any(not 0.0 <= xi <= 1.0 for xi in args.slice_xi):
        raise ValueError("slice-xi values must lie in [0, 1]")

    p = CathodeParameters()
    vcell_v = p.tech.bol_typical_cell_voltage_v(args.current_a)
    heat_rejection_w = p.stack.heat_rejection_w(args.current_a, vcell_v)
    target_temperature_c = p.tech.optimum_stack_temperature_c(args.current_a)
    target_temperature_k = 273.15 + target_temperature_c
    inlet_temperature_k = 273.15 + args.inlet_temperature_c
    floor_flow_slpm = p.stack.coolant_air_target_slpm(args.current_a)
    airflow = open_cathode_airflow_target(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        target_stack_temperature_k=target_temperature_k,
        stoichiometric_floor_slpm=floor_flow_slpm,
    )
    if airflow.target_air_flow_slpm is None:
        raise RuntimeError("selected operating point is thermally unreachable")

    profile = solve_streamwise_finite_thermal_profile(
        current_a=args.current_a,
        total_air_flow_slpm=airflow.target_air_flow_slpm,
        n_cells=p.stack.n_cells,
        oxygen_mole_fraction=p.oxygen_mole_fraction,
        faraday_c_mol=p.faraday,
        inlet_temperature_k=inlet_temperature_k,
        stack_temperature_k=target_temperature_k,
        ntu=args.ntu,
        points=args.profile_points,
    )
    concentration = ideal_gas_species_concentration_mol_m3(
        mole_fraction=profile.oxygen_mole_fraction,
        pressure_pa=p.pressure,
        temperature_k=profile.air_temperature_k,
        gas_constant_j_mol_k=p.gas_constant,
    )

    rows: list[dict[str, Any]] = []
    for xi in args.slice_xi:
        local_c = float(
            np.interp(
                xi,
                profile.streamwise_fraction,
                concentration,
            )
        )
        slice_dir = args.work_dir / f"xi-{xi:.3f}"
        if slice_dir.exists():
            shutil.rmtree(slice_dir)

        run_cathode_electrochem(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            stop_time=args.stop_time,
            max_dt=args.max_dt,
            output_dir=slice_dir,
            scalar_dt=max(args.stop_time / 20.0, 1.0e-7),
            oxygen_feed_concentration=local_c,
        )

        rows.append(
            {
                "slice_xi": xi,
                "oxygen_feed_concentration_mol_m3": local_c,
                "oxygen_activity_feed": local_c / p.oxygen_inlet_concentration,
                "mean_o2_concentration_mol_m3": _last_scalar(
                    slice_dir,
                    "mean_c_o2",
                ),
                "mean_j_orr_vol_a_m3": _last_scalar(
                    slice_dir,
                    "mean_j_orr_vol",
                ),
                "total_reaction_current": _last_scalar(
                    slice_dir,
                    "total_reaction_current",
                ),
                "mean_eta_v": _last_scalar(slice_dir, "mean_eta"),
            }
        )

    inlet_current = float(rows[0]["total_reaction_current"])
    for row in rows:
        row["reaction_current_relative_to_inlet"] = (
            float(row["total_reaction_current"]) / inlet_current
        )

    current_values = [float(row["total_reaction_current"]) for row in rows]
    relative_span = (max(current_values) - min(current_values)) / max(current_values)

    output = {
        "schema_version": 1,
        "model": "v09-quasi3d-electrochemical-slices",
        "current_a": args.current_a,
        "inlet_temperature_c": args.inlet_temperature_c,
        "ntu": args.ntu,
        "grid": [args.nx, args.ny, args.nz],
        "stop_time": args.stop_time,
        "relative_reaction_current_span": relative_span,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"slice_count={len(rows)} "
        f"relative_reaction_current_span={100.0 * relative_span:.3f}%",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
