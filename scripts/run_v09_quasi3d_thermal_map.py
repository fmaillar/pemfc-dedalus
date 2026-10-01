"""Map nominal-current quasi-3D ORR sensitivity over Tin and NTU."""

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
    parser.add_argument(
        "--inlet-temperatures-c",
        nargs="+",
        type=float,
        default=[10.0, 20.0, 30.0],
    )
    parser.add_argument(
        "--ntu-values",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0],
    )
    parser.add_argument("--profile-points", type=int, default=64)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--stop-time", type=float, default=1.0e-3)
    parser.add_argument("--max-dt", type=float, default=2.0e-6)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output-v09-quasi3d-thermal-map"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quick-v09-quasi3d-thermal-map.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick-v09-quasi3d-thermal-map.csv"),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    vcell_v = p.tech.bol_typical_cell_voltage_v(args.current_a)
    heat_rejection_w = p.stack.heat_rejection_w(args.current_a, vcell_v)
    target_temperature_c = p.tech.optimum_stack_temperature_c(args.current_a)
    target_temperature_k = 273.15 + target_temperature_c
    floor_flow_slpm = p.stack.coolant_air_target_slpm(args.current_a)

    rows: list[dict[str, Any]] = []
    case_spans: list[float] = []

    for inlet_temperature_c in args.inlet_temperatures_c:
        inlet_temperature_k = 273.15 + inlet_temperature_c
        airflow = open_cathode_airflow_target(
            heat_rejection_w=heat_rejection_w,
            inlet_temperature_k=inlet_temperature_k,
            target_stack_temperature_k=target_temperature_k,
            stoichiometric_floor_slpm=floor_flow_slpm,
        )
        if airflow.target_air_flow_slpm is None:
            continue

        for ntu in args.ntu_values:
            profile = solve_streamwise_finite_thermal_profile(
                current_a=args.current_a,
                total_air_flow_slpm=airflow.target_air_flow_slpm,
                n_cells=p.stack.n_cells,
                oxygen_mole_fraction=p.oxygen_mole_fraction,
                faraday_c_mol=p.faraday,
                inlet_temperature_k=inlet_temperature_k,
                stack_temperature_k=target_temperature_k,
                ntu=ntu,
                points=args.profile_points,
            )
            concentration = ideal_gas_species_concentration_mol_m3(
                mole_fraction=profile.oxygen_mole_fraction,
                pressure_pa=p.pressure,
                temperature_k=profile.air_temperature_k,
                gas_constant_j_mol_k=p.gas_constant,
            )
            local_feed = {
                xi: float(np.interp(xi, profile.streamwise_fraction, concentration))
                for xi in (0.0, 1.0)
            }

            currents: dict[float, float] = {}
            case_rows: list[dict[str, Any]] = []
            for xi in (0.0, 1.0):
                run_dir = (
                    args.work_dir
                    / f"tin-{inlet_temperature_c:.1f}"
                    / f"ntu-{ntu:.1f}"
                    / f"xi-{xi:.1f}"
                )
                if run_dir.exists():
                    shutil.rmtree(run_dir)

                run_cathode_electrochem(
                    nx=args.nx,
                    ny=args.ny,
                    nz=args.nz,
                    stop_time=args.stop_time,
                    max_dt=args.max_dt,
                    output_dir=run_dir,
                    scalar_dt=max(args.stop_time / 20.0, 1.0e-7),
                    oxygen_feed_concentration=local_feed[xi],
                )

                current = _last_scalar(run_dir, "total_reaction_current")
                currents[xi] = current
                row = {
                    "current_a": args.current_a,
                    "inlet_temperature_c": inlet_temperature_c,
                    "ntu": ntu,
                    "slice_xi": xi,
                    "oxygen_feed_concentration_mol_m3": local_feed[xi],
                    "total_reaction_current": current,
                    "mean_j_orr_vol_a_m3": _last_scalar(
                        run_dir,
                        "mean_j_orr_vol",
                    ),
                    "mean_eta_v": _last_scalar(run_dir, "mean_eta"),
                }
                case_rows.append(row)

            span = (currents[0.0] - currents[1.0]) / currents[0.0]
            case_spans.append(span)
            for row in case_rows:
                row["reaction_current_span"] = span
                rows.append(row)

    output = {
        "schema_version": 1,
        "model": "v09-quasi3d-thermal-map",
        "current_a": args.current_a,
        "grid": [args.nx, args.ny, args.nz],
        "stop_time": args.stop_time,
        "span_min": min(case_spans),
        "span_max": max(case_spans),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"cases={len(case_spans)} "
        f"span_min={100.0 * min(case_spans):.3f}% "
        f"span_max={100.0 * max(case_spans):.3f}%",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
