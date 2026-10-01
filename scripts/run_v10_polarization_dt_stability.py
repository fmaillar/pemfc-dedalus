"""Test timestep sensitivity of the late V1.0 polarization instability."""

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
        raise FileNotFoundError(f"no scalar files found in {path / 'scalars'}")
    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            times.append(np.asarray(handle["scales/sim_time"]).reshape(-1))
            raw = np.asarray(handle[f"tasks/{task}"])
            values.append(raw.reshape(raw.shape[0], -1)[:, 0])
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
    times, reaction = _scalar_series(run_dir, "total_reaction_current")
    area_m2 = params.length_x * params.length_y
    return {
        "slice_xi": xi,
        "time_s": times.tolist(),
        "current_density_a_m2": (reaction / area_m2).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solutions-json",
        type=Path,
        default=Path("results/quick-v10-polarization.json"),
    )
    parser.add_argument("--current-a", type=float, default=26.04)
    parser.add_argument(
        "--max-dts",
        nargs="+",
        type=float,
        default=[2e-6, 1e-6, 5e-7],
    )
    parser.add_argument("--inlet-temperature-c", type=float, default=20.0)
    parser.add_argument("--ntu", type=float, default=3.0)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--stop-time", type=float, default=8e-3)
    parser.add_argument("--scalar-dt", type=float, default=1e-4)
    parser.add_argument("--profile-points", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output-v10-polarization-dt-stability"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v10-polarization-dt-stability.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v10-polarization-dt-stability.csv"),
    )
    args = parser.parse_args()

    data = json.loads(args.solutions_json.read_text())
    solution = next(
        item
        for item in data["solutions"]
        if float(item["current_a"]) == args.current_a
    )
    voltage_v = float(solution["solution_voltage_v"])

    base = CathodeParameters()
    target_temperature_k = (
        273.15 + base.tech.optimum_stack_temperature_c(args.current_a)
    )
    inlet_temperature_k = 273.15 + args.inlet_temperature_c
    params = replace(
        base,
        stack_current_a=args.current_a,
        stack_temperature=target_temperature_k,
        cathode_solid_potential=voltage_v,
    )

    heat_rejection_w = params.stack.heat_rejection_w(args.current_a, voltage_v)
    airflow = open_cathode_airflow_target(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        target_stack_temperature_k=target_temperature_k,
        stoichiometric_floor_slpm=params.stack.coolant_air_target_slpm(args.current_a),
    )
    if airflow.target_air_flow_slpm is None:
        raise RuntimeError("selected operating point is thermally unreachable")

    profile = solve_streamwise_finite_thermal_profile(
        current_a=args.current_a,
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
    xis = [0.0, 0.5, 1.0]
    local_feed = {
        xi: float(np.interp(xi, profile.streamwise_fraction, concentration))
        for xi in xis
    }

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for max_dt in args.max_dts:
        with ProcessPoolExecutor(max_workers=min(args.jobs, 3)) as executor:
            futures = []
            for xi in xis:
                futures.append(
                    executor.submit(
                        _run_slice,
                        params=params,
                        xi=xi,
                        oxygen_feed_concentration=local_feed[xi],
                        run_dir=(
                            args.work_dir
                            / f"dt-{max_dt:.0e}"
                            / f"xi-{xi:.1f}"
                        ),
                        nx=args.nx,
                        ny=args.ny,
                        nz=args.nz,
                        stop_time=args.stop_time,
                        max_dt=max_dt,
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
        mask = times >= 2e-3
        post_times = times[mask]
        post_current = mean_current[mask]
        min_index = int(np.argmin(post_current))
        minimum = float(post_current[min_index])
        maximum = float(np.max(post_current))
        final = float(mean_current[-1])

        summary = {
            "max_dt": max_dt,
            "minimum_after_2ms_a_m2": minimum,
            "maximum_after_2ms_a_m2": maximum,
            "time_of_minimum_s": float(post_times[min_index]),
            "final_current_density_a_m2": final,
            "final_to_minimum_ratio": final / minimum,
        }
        summaries.append(summary)

        for index, time_s in enumerate(times):
            rows.append(
                {
                    "max_dt": max_dt,
                    "time_s": float(time_s),
                    "mean_current_density_a_m2": float(mean_current[index]),
                    "j_xi_0_a_m2": float(currents[0, index]),
                    "j_xi_05_a_m2": float(currents[1, index]),
                    "j_xi_1_a_m2": float(currents[2, index]),
                }
            )

    output = {
        "schema_version": 1,
        "model": "v10-polarization-timestep-stability",
        "current_a": args.current_a,
        "voltage_v": voltage_v,
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for item in summaries:
        print(
            f"dt={item['max_dt']:.1e} "
            f"t_min={item['time_of_minimum_s']:.4e} "
            f"j_min={item['minimum_after_2ms_a_m2']:.2f} "
            f"j_final={item['final_current_density_a_m2']:.2f} "
            f"ratio={item['final_to_minimum_ratio']:.4f}",
            flush=True,
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
