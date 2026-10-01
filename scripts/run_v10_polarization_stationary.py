"""Diagnose stationary windows in the V1.0 electrochemical pseudo-transient."""

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


def _scalar_series(path: Path, task: str) -> tuple[np.ndarray, np.ndarray]:
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    files = sorted((path / "scalars").glob("scalars_s*.h5"))
    if not files:
        raise FileNotFoundError(f"no Dedalus scalar files found in {path / 'scalars'}")

    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            times.append(np.asarray(handle["scales/sim_time"]).reshape(-1))
            task_values = np.asarray(handle[f"tasks/{task}"])
            values.append(task_values.reshape(task_values.shape[0], -1)[:, 0])

    return np.concatenate(times), np.concatenate(values)


def simpson_mean(values: np.ndarray) -> np.ndarray:
    if values.shape[0] != 3:
        raise ValueError("Simpson mean requires exactly three rows")
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
    scalar_dt: float,
) -> dict[str, Any]:
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
        scalar_dt=scalar_dt,
        oxygen_feed_concentration=oxygen_feed_concentration,
    )

    times, reaction_current = _scalar_series(run_dir, "total_reaction_current")
    eta_times, eta = _scalar_series(run_dir, "mean_eta")
    if not np.allclose(times, eta_times):
        raise RuntimeError("scalar time bases differ")

    area_m2 = params.length_x * params.length_y
    return {
        "slice_xi": xi,
        "time_s": times.tolist(),
        "current_density_a_m2": (reaction_current / area_m2).tolist(),
        "mean_eta_v": eta.tolist(),
    }


def _find_stable_window(
    times: np.ndarray,
    values: np.ndarray,
    *,
    window_points: int,
    relative_tolerance: float,
    skip_fraction: float,
) -> dict[str, Any] | None:
    if window_points < 2:
        raise ValueError("window_points must be at least 2")
    start_index = max(0, int(skip_fraction * len(values)))

    for end in range(start_index + window_points, len(values) + 1):
        begin = end - window_points
        window = values[begin:end]
        mean = float(np.mean(window))
        if mean == 0.0:
            continue
        relative_span = float((np.max(window) - np.min(window)) / abs(mean))
        if relative_span <= relative_tolerance:
            return {
                "start_time_s": float(times[begin]),
                "end_time_s": float(times[end - 1]),
                "mean_current_density_a_m2": mean,
                "relative_span": relative_span,
                "window_points": window_points,
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solutions-json",
        type=Path,
        default=Path("results/quick-v10-polarization.json"),
    )
    parser.add_argument("--currents-a", nargs="+", type=float, default=[7.3, 26.04])
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--ntu", type=float, default=3.0)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--stop-time", type=float, default=8e-3)
    parser.add_argument("--max-dt", type=float, default=2e-6)
    parser.add_argument("--scalar-dt", type=float, default=1e-4)
    parser.add_argument("--window-points", type=int, default=6)
    parser.add_argument("--relative-tolerance", type=float, default=5e-3)
    parser.add_argument("--skip-fraction", type=float, default=0.25)
    parser.add_argument("--profile-points", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output-v10-polarization-stationary"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v10-polarization-stationary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v10-polarization-stationary.csv"),
    )
    args = parser.parse_args()

    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    if not 0.0 <= args.skip_fraction < 1.0:
        raise ValueError("skip-fraction must be in [0, 1)")

    solution_data = json.loads(args.solutions_json.read_text())
    voltage_by_current = {
        float(item["current_a"]): float(item["solution_voltage_v"])
        for item in solution_data["solutions"]
    }

    base = CathodeParameters()
    inlet_temperature_k = 273.15 + args.inlet_temperature_c
    xis = [0.0, 0.5, 1.0]
    output_cases: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for current_a in args.currents_a:
        voltage_v = voltage_by_current[current_a]
        target_temperature_k = (
            273.15 + base.tech.optimum_stack_temperature_c(current_a)
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

        with ProcessPoolExecutor(max_workers=min(args.jobs, 3)) as executor:
            futures = []
            for xi in xis:
                run_dir = (
                    args.work_dir
                    / f"current-{current_a:.2f}"
                    / f"xi-{xi:.1f}"
                )
                futures.append(
                    executor.submit(
                        _run_slice,
                        params=params,
                        xi=xi,
                        oxygen_feed_concentration=local_feed[xi],
                        run_dir=run_dir,
                        nx=args.nx,
                        ny=args.ny,
                        nz=args.nz,
                        stop_time=args.stop_time,
                        max_dt=args.max_dt,
                        scalar_dt=args.scalar_dt,
                    )
                )
            slices = sorted(
                (future.result() for future in futures),
                key=lambda item: item["slice_xi"],
            )

        times = np.asarray(slices[0]["time_s"], dtype=float)
        currents = np.vstack(
            [np.asarray(item["current_density_a_m2"], dtype=float) for item in slices]
        )
        if any(
            not np.allclose(times, np.asarray(item["time_s"], dtype=float))
            for item in slices[1:]
        ):
            raise RuntimeError("slice time bases differ")

        mean_current = simpson_mean(currents)
        stable_window = _find_stable_window(
            times,
            mean_current,
            window_points=args.window_points,
            relative_tolerance=args.relative_tolerance,
            skip_fraction=args.skip_fraction,
        )
        after_skip = mean_current[int(args.skip_fraction * len(mean_current)) :]
        case = {
            "current_a": current_a,
            "voltage_v": voltage_v,
            "stable_window": stable_window,
            "post_skip_min_current_density_a_m2": float(np.min(after_skip)),
            "post_skip_max_current_density_a_m2": float(np.max(after_skip)),
            "final_current_density_a_m2": float(mean_current[-1]),
        }
        output_cases.append(case)

        for index, time_s in enumerate(times):
            csv_rows.append(
                {
                    "current_a": current_a,
                    "voltage_v": voltage_v,
                    "time_s": float(time_s),
                    "mean_current_density_a_m2": float(mean_current[index]),
                    "j_xi_0_a_m2": float(currents[0, index]),
                    "j_xi_05_a_m2": float(currents[1, index]),
                    "j_xi_1_a_m2": float(currents[2, index]),
                }
            )

    output = {
        "schema_version": 1,
        "model": "v10-polarization-stationary-diagnostic",
        "grid": [args.nx, args.ny, args.nz],
        "stop_time": args.stop_time,
        "scalar_dt": args.scalar_dt,
        "window_points": args.window_points,
        "relative_tolerance": args.relative_tolerance,
        "cases": output_cases,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    for case in output_cases:
        print(
            f"I={case['current_a']:.2f} A "
            f"stable_window={case['stable_window']} "
            f"final_j={case['final_current_density_a_m2']:.2f} A/m2",
            flush=True,
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
