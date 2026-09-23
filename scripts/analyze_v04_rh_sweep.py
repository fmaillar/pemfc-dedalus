"""Analyze the completed V0.4 RH-voltage sweep.

Reads the consolidated overnight JSON and writes:
- a long-form CSV with derived current-density and logarithmic sensitivities,
- a compact JSON summary by cathode voltage,
- publication-friendly CSV tables for polarization and RH sensitivity.

No Dedalus solve is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def central_derivative(xs: list[float], ys: list[float]) -> list[float]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return [math.nan] * len(xs)

    out: list[float] = []
    for i in range(len(xs)):
        if i == 0:
            dx = xs[1] - xs[0]
            out.append((ys[1] - ys[0]) / dx)
        elif i == len(xs) - 1:
            dx = xs[-1] - xs[-2]
            out.append((ys[-1] - ys[-2]) / dx)
        else:
            dx = xs[i + 1] - xs[i - 1]
            out.append((ys[i + 1] - ys[i - 1]) / dx)
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/overnight-v04-rh-voltage.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis-v04-rh"),
    )
    args = parser.parse_args()

    study = json.loads(args.input.read_text())
    cases = list(study["cases"])
    if not cases:
        raise SystemExit("no cases found")

    # Representative patch area is the x-y footprint used by all V0.4 cases.
    # Keep it explicit here so current-density values cannot be mistaken for stack current.
    length_x_m = 2.0e-3
    length_y_m = 1.0e-3
    area_m2 = length_x_m * length_y_m

    voltages = sorted({float(c["cathode_solid_potential_v"]) for c in cases})
    rhs = sorted({float(c["rh"]) for c in cases})

    long_rows: list[dict[str, Any]] = []
    summary_by_voltage: dict[str, Any] = {}
    polarization_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for voltage in voltages:
        vcases = sorted(
            (c for c in cases if float(c["cathode_solid_potential_v"]) == voltage),
            key=lambda c: float(c["rh"]),
        )
        vrh = [float(c["rh"]) for c in vcases]
        currents = [float(c["total_reaction_current_a"]) for c in vcases]
        sigmas = [float(c["sigma_m_max_s_per_m"]) for c in vcases]

        dI_dRH = central_derivative(vrh, currents)
        dlnI_dlnsigma = central_derivative(
            [math.log(s) for s in sigmas],
            [math.log(i) for i in currents],
        )

        for c, sens_rh, elasticity in zip(vcases, dI_dRH, dlnI_dlnsigma, strict=True):
            current = float(c["total_reaction_current_a"])
            current_density = current / area_m2
            row = {
                "rh": float(c["rh"]),
                "rh_percent": 100.0 * float(c["rh"]),
                "cathode_solid_potential_v": voltage,
                "total_reaction_current_a": current,
                "current_density_a_per_m2": current_density,
                "current_density_a_per_cm2": current_density / 1.0e4,
                "sigma_m_max_s_per_m": float(c["sigma_m_max_s_per_m"]),
                "lambda_cl_max": float(c["lambda_cl_max"]),
                "mean_c_o2_mol_per_m3": float(c["mean_c_o2_mol_per_m3"]),
                "mean_eta_v": float(c["mean_eta_v"]),
                "phi_m_min_v": float(c["phi_m_min_v"]),
                "dI_dRH_a_per_fraction": sens_rh,
                "dI_dRH_ma_per_percent": sens_rh * 10.0,
                "dlnI_dlnsigma": elasticity,
            }
            long_rows.append(row)
            polarization_rows.append(
                {
                    "rh_percent": row["rh_percent"],
                    "cathode_solid_potential_v": voltage,
                    "current_density_a_per_cm2": row["current_density_a_per_cm2"],
                }
            )
            sensitivity_rows.append(
                {
                    "rh_percent": row["rh_percent"],
                    "cathode_solid_potential_v": voltage,
                    "dI_dRH_ma_per_percent": row["dI_dRH_ma_per_percent"],
                    "dlnI_dlnsigma": elasticity,
                }
            )

        i_low = currents[0]
        i_high = currents[-1]
        summary_by_voltage[f"{voltage:.3f}"] = {
            "current_rh10_a": i_low,
            "current_rh90_a": i_high,
            "relative_gain_rh10_to_rh90": (i_high - i_low) / i_low,
            "max_abs_dI_dRH_a_per_fraction": max(abs(x) for x in dI_dRH),
            "max_dlnI_dlnsigma": max(dlnI_dlnsigma),
            "min_dlnI_dlnsigma": min(dlnI_dlnsigma),
        }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    write_csv(
        out / "derived.csv",
        [
            "rh",
            "rh_percent",
            "cathode_solid_potential_v",
            "total_reaction_current_a",
            "current_density_a_per_m2",
            "current_density_a_per_cm2",
            "sigma_m_max_s_per_m",
            "lambda_cl_max",
            "mean_c_o2_mol_per_m3",
            "mean_eta_v",
            "phi_m_min_v",
            "dI_dRH_a_per_fraction",
            "dI_dRH_ma_per_percent",
            "dlnI_dlnsigma",
        ],
        long_rows,
    )
    write_csv(
        out / "polarization.csv",
        ["rh_percent", "cathode_solid_potential_v", "current_density_a_per_cm2"],
        polarization_rows,
    )
    write_csv(
        out / "sensitivity.csv",
        [
            "rh_percent",
            "cathode_solid_potential_v",
            "dI_dRH_ma_per_percent",
            "dlnI_dlnsigma",
        ],
        sensitivity_rows,
    )

    summary = {
        "schema_version": 1,
        "source": str(args.input),
        "grid": study.get("grid"),
        "mpi_ranks": study.get("mpi_ranks"),
        "representative_area_m2": area_m2,
        "relative_humidities": rhs,
        "cathode_solid_potentials_v": voltages,
        "summary_by_voltage": summary_by_voltage,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {out / 'derived.csv'}")
    print(f"Wrote {out / 'polarization.csv'}")
    print(f"Wrote {out / 'sensitivity.csv'}")
    print(f"Wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
