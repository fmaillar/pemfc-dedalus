"""Analyze and plot the converged V11 reference trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _purge_times(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [
            float(row["time_s"])
            for row in rows
            if row["purge_event"] == "True"
        ],
        dtype=float,
    )


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/v11-reference/v11-reference-dt-0p005.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v11-reference/figures"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v11-reference-analysis.json"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.input_csv)
    if not rows:
        raise ValueError("trajectory CSV is empty")

    time_s = _array(rows, "time_s")
    current_a = _array(rows, "current_a")
    airflow_slpm = _array(rows, "stack_air_flow_slpm")
    voltage_v = _array(rows, "cell_voltage_v")
    temperature_k = _array(rows, "stack_temperature_k")
    lambda_mem = _array(rows, "membrane_mean_water_content")
    anode_activity = _array(rows, "anode_water_activity")
    cathode_activity = _array(rows, "cathode_water_activity")
    anode_n2 = _array(rows, "anode_nitrogen_mol")
    n2_crossover = _array(rows, "nitrogen_crossover_rate_mol_s")
    purge_times = _purge_times(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_s, voltage_v, label="Cell voltage [V]")
    for purge_time in purge_times:
        plt.axvline(purge_time, linestyle="--", linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Cell voltage [V]")
    plt.grid(True, alpha=0.3)
    _save_figure(args.output_dir / "voltage.png")

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_s, temperature_k - 273.15)
    for purge_time in purge_times:
        plt.axvline(purge_time, linestyle="--", linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Stack temperature [degC]")
    plt.grid(True, alpha=0.3)
    _save_figure(args.output_dir / "temperature.png")

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_s, lambda_mem, label="Membrane lambda")
    plt.plot(time_s, anode_activity, label="Anode water activity")
    plt.plot(time_s, cathode_activity, label="Cathode water activity")
    for purge_time in purge_times:
        plt.axvline(purge_time, linestyle="--", linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Hydration / water activity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(args.output_dir / "hydration.png")

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_s, anode_n2 * 1.0e6, label="Anode N2 [umol]")
    for purge_time in purge_times:
        plt.axvline(purge_time, linestyle="--", linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Anode N2 [umol]")
    plt.grid(True, alpha=0.3)
    _save_figure(args.output_dir / "anode-nitrogen.png")

    plt.figure(figsize=(9, 4.5))
    plt.step(time_s, current_a, where="post", label="Current [A]")
    plt.step(time_s, airflow_slpm / 10.0, where="post", label="Airflow / 10")
    plt.xlabel("Time [s]")
    plt.ylabel("Control magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure(args.output_dir / "controls.png")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": "v11-reference-analysis",
        "source_csv": str(args.input_csv),
        "purge_times_s": purge_times.tolist(),
        "purge_count": int(purge_times.size),
        "cell_voltage_v": {
            "min": float(np.min(voltage_v)),
            "max": float(np.max(voltage_v)),
            "final": float(voltage_v[-1]),
        },
        "stack_temperature_k": {
            "min": float(np.min(temperature_k)),
            "max": float(np.max(temperature_k)),
            "final": float(temperature_k[-1]),
        },
        "membrane_mean_water_content": {
            "min": float(np.min(lambda_mem)),
            "max": float(np.max(lambda_mem)),
            "final": float(lambda_mem[-1]),
        },
        "anode_water_activity": {
            "min": float(np.min(anode_activity)),
            "max": float(np.max(anode_activity)),
            "final": float(anode_activity[-1]),
        },
        "cathode_water_activity": {
            "min": float(np.min(cathode_activity)),
            "max": float(np.max(cathode_activity)),
            "final": float(cathode_activity[-1]),
        },
        "anode_nitrogen_mol": {
            "min": float(np.min(anode_n2)),
            "max": float(np.max(anode_n2)),
            "final": float(anode_n2[-1]),
        },
        "nitrogen_crossover_rate_mol_s": {
            "min": float(np.min(n2_crossover)),
            "max": float(np.max(n2_crossover)),
            "final": float(n2_crossover[-1]),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"purges={summary['purge_count']} "
        f"V=[{summary['cell_voltage_v']['min']:.6f}, "
        f"{summary['cell_voltage_v']['max']:.6f}] V "
        f"Tmax={summary['stack_temperature_k']['max'] - 273.15:.3f} degC "
        f"lambda_final={summary['membrane_mean_water_content']['final']:.4f}",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote figures in {args.output_dir}")


if __name__ == "__main__":
    main()
