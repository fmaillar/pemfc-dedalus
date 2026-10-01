"""Extend pseudo-time convergence checks for V1.0 polarization solutions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
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


def _run_slice(
    *,
    params: CathodeParameters,
    xi: float,
    oxygen_feed_concentration: float,
    run_dir: Path,
    nx: int,
    ny: int,
    nz: int,
    stop_time: float,
    max_dt: float,
) -> dict[str, float]:
    if run_dir.exists():
        shutil.rmtree(run_dir)

    run_cathode_electrochem(
        params=params,
        nx=nx,
        ny=ny,
        nz=nz,
        stop_time=stop_time,
        max_dt=max_dt,
        output_dir=run_dir,
        scalar_dt=max(stop_time / 40.0, 1.0e-7),
        oxygen_feed_concentration=oxygen_feed_concentration,
    )
    area_m2 = params.length_x * params.length_y
    reaction_current = _last_scalar(run_dir, "total_reaction_current")
    return {
        "slice_xi": xi,
        "current_density_a_m2": reaction_current / area_m2,
        "mean_eta_v": _last_scalar(run_dir, "mean_eta"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solutions-json",
        type=Path,
        default=Path("results/quick-v10-polarization.json"),
    )
    parser.add_argument("--currents-a", nargs="+", type=float, default=[7.3, 26.04])
    parser.add_argument(
        "--stop-times",
        nargs="+",
        type=float,
        default=[1e-3, 2e-3, 4e-3, 8e-3],
    )
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--ntu", type=float, default=3.0)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--profile-points", type=int, default=64)
    parser.add_argument("--max-dt", type=float, default=2e-6)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output-v10-polarization-time-convergence"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v10-polarization-time-convergence.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v10-polarization-time-convergence.csv"),
    )
    args = parser.parse_args()

    if args.jobs <= 0:
        raise ValueError("jobs must be positive")

    solution_data = json.loads(args.solutions_json.read_text())
    voltage_by_current = {
        float(item["current_a"]): float(item["solution_voltage_v"])
        for item in solution_data["solutions"]
    }

    base = CathodeParameters()
    reference_current_a = base.stack_current_a
    inlet_temperature_k = 273.15 + args.inlet_temperature_c
    xis = [0.0, 0.5, 1.0]
    tasks: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    for current_a in args.currents_a:
        voltage_v = voltage_by_current[current_a]
        target_temperature_k = (
            273.15 + base.tech.optimum_stack_temperature_c(current_a)
        )
        target_current_density = (
            base.membrane_current_density * current_a / reference_current_a
        )
        params = replace(
            base,
            stack_current_a=current_a,
            stack_temperature=target_temperature_k,
            cathode_solid_potential=voltage_v,
        )
        heat_rejection_w = params.stack.heat_rejection_w(current_a, voltage_v)
        airflow = open_cathode_airflow_target(
            heat_rejection_w=heat_rejection_w,
            inlet_temperature_k=inlet_temperature_k,
            target_stack_temperature_k=target_temperature_k,
            stoichiometric_floor_slpm=params.stack.coolant_air_target_slpm(current_a),
        )
        if airflow.target_air_flow_slpm is None:
            raise RuntimeError(f"thermally unreachable current={current_a} A")

        profile = solve_streamwise_finite_thermal_profile(
            current_a=current_a,
            total_air_flow_slpm=airflow.target_air_flow_slpm,
            n_cells=params.stack.n_cells,
            oxygen_mole_fraction=params.oxygen_mole_fraction,
            faraday_c_mol=params.faraday,
            inlet_temperature_k=inlet_temperature_k,
            stack_temperature_k=target_temperature_k,
            ntu=args.ntu,
            points=args.profile_points,
        )
        concentration = ideal_gas_species_concentration_mol_m3(
            mole_fraction=profile.oxygen_mole_fraction,
            pressure_pa=params.pressure,
            temperature_k=profile.air_temperature_k,
            gas_constant_j_mol_k=params.gas_constant,
        )
        local_feed = {
            xi: float(np.interp(xi, profile.streamwise_fraction, concentration))
            for xi in xis
        }

        for stop_time in args.stop_times:
            key = f"{current_a:.2f}|{stop_time:.6g}"
            metadata[key] = {
                "current_a": current_a,
                "voltage_v": voltage_v,
                "target_current_density_a_m2": target_current_density,
                "stop_time": stop_time,
                "slice_rows": [],
            }
            for xi in xis:
                tasks.append(
                    {
                        "key": key,
                        "params": params,
                        "xi": xi,
                        "oxygen_feed_concentration": local_feed[xi],
                        "run_dir": (
                            args.work_dir
                            / f"current-{current_a:.2f}"
                            / f"time-{stop_time:.0e}"
                            / f"xi-{xi:.1f}"
                        ),
                        "stop_time": stop_time,
                    }
                )

    with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as executor:
        pending = []
        for task in tasks:
            future = executor.submit(
                _run_slice,
                params=task["params"],
                xi=task["xi"],
                oxygen_feed_concentration=task["oxygen_feed_concentration"],
                run_dir=task["run_dir"],
                nx=args.nx,
                ny=args.ny,
                nz=args.nz,
                stop_time=task["stop_time"],
                max_dt=args.max_dt,
            )
            pending.append((task["key"], future))
        for key, future in pending:
            metadata[key]["slice_rows"].append(future.result())

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for item in metadata.values():
        slice_rows = sorted(item["slice_rows"], key=lambda row: row["slice_xi"])
        densities = [float(row["current_density_a_m2"]) for row in slice_rows]
        mean_density = simpson_mean(densities)
        target = float(item["target_current_density_a_m2"])
        summary = {
            "current_a": item["current_a"],
            "voltage_v": item["voltage_v"],
            "stop_time": item["stop_time"],
            "mean_current_density_a_m2": mean_density,
            "target_current_density_a_m2": target,
            "relative_error": (mean_density - target) / target,
        }
        summaries.append(summary)
        for row in slice_rows:
            rows.append(
                {
                    **summary,
                    "slice_xi": row["slice_xi"],
                    "local_current_density_a_m2": row["current_density_a_m2"],
                    "mean_eta_v": row["mean_eta_v"],
                }
            )

    changes: list[dict[str, float]] = []
    for current_a in args.currents_a:
        current_rows = sorted(
            (
                item
                for item in summaries
                if float(item["current_a"]) == current_a
            ),
            key=lambda item: float(item["stop_time"]),
        )
        for previous, current in zip(current_rows, current_rows[1:], strict=False):
            previous_j = float(previous["mean_current_density_a_m2"])
            current_j = float(current["mean_current_density_a_m2"])
            changes.append(
                {
                    "current_a": current_a,
                    "from_stop_time": float(previous["stop_time"]),
                    "to_stop_time": float(current["stop_time"]),
                    "relative_change": (current_j - previous_j) / previous_j,
                }
            )

    output = {
        "schema_version": 1,
        "model": "v10-polarization-pseudotime-convergence",
        "grid": [args.nx, args.ny, args.nz],
        "summaries": summaries,
        "successive_changes": changes,
        "max_abs_last_step_change": max(
            abs(item["relative_change"])
            for item in changes
            if item["to_stop_time"] == max(args.stop_times)
        ),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for item in sorted(
        summaries,
        key=lambda row: (float(row["current_a"]), float(row["stop_time"])),
    ):
        print(
            f"I={item['current_a']:.2f} A "
            f"t={item['stop_time']:.1e} "
            f"j={item['mean_current_density_a_m2']:.2f} A/m2 "
            f"error={100.0 * item['relative_error']:+.3f}%",
            flush=True,
        )
    for item in changes:
        print(
            f"I={item['current_a']:.2f} A "
            f"{item['from_stop_time']:.1e}->{item['to_stop_time']:.1e}: "
            f"dj={100.0 * item['relative_change']:+.3f}%",
            flush=True,
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
