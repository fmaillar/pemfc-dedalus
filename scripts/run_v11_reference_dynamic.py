"""Run the V11 reference transient and a parallel time-step convergence study."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.v11_reference import reference_dynamic_scenario
from pemfc_dedalus.v11_runner import run_v11_dynamic, trajectory_to_rows

DEFAULT_DT_VALUES_S = (0.04, 0.02, 0.01, 0.005)
COMPARISON_FIELDS = (
    "cell_voltage_v",
    "stack_temperature_k",
    "membrane_mean_water_content",
    "anode_nitrogen_mol",
)


def _write_csv(
    path: Path,
    rows: list[dict[str, float | int | bool]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty V11 trajectory")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_one_case(
    dt_s: float,
    sample_every_s: float,
    output_dir: str,
    stop_time_s: float,
) -> dict[str, Any]:
    try:
        scenario = reference_dynamic_scenario()
        effective_stop = min(stop_time_s, scenario.stop_time_s)
        trajectory = run_v11_dynamic(
            initial_state=scenario.initial_state,
            controls=scenario.controls,
            inputs=scenario.inputs,
            stop_time_s=effective_stop,
            dt_s=dt_s,
            automatic_purge=True,
            sample_every_s=sample_every_s,
        )
        rows = trajectory_to_rows(trajectory)
        dt_label = f"{dt_s:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        csv_path = Path(output_dir) / f"v11-reference-dt-{dt_label}.csv"
        _write_csv(csv_path, rows)

        final = rows[-1]
        summary = {
            "status": "ok",
            "error": "",
            "dt_s": dt_s,
            "rows": len(rows),
            "csv_path": str(csv_path),
            "purge_count": int(final["purge_count"]),
            "final_cell_voltage_v": float(final["cell_voltage_v"]),
            "final_stack_temperature_k": float(final["stack_temperature_k"]),
            "final_membrane_mean_water_content": float(
                final["membrane_mean_water_content"]
            ),
            "final_anode_nitrogen_mol": float(final["anode_nitrogen_mol"]),
        }
        summary_path = Path(output_dir) / f"v11-reference-dt-{dt_label}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "dt_s": dt_s,
            "rows": 0,
            "csv_path": "",
            "purge_count": 0,
            "final_cell_voltage_v": None,
            "final_stack_temperature_k": None,
            "final_membrane_mean_water_content": None,
            "final_anode_nitrogen_mol": None,
        }


def _read_numeric_column(
    rows: list[dict[str, str]],
    field: str,
) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_time_series(
    rows: list[dict[str, str]],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    times = _read_numeric_column(rows, "time_s")
    values = _read_numeric_column(rows, field)
    unique_times, unique_indices = np.unique(times, return_index=True)
    return unique_times, values[unique_indices]


def convergence_metrics(
    *,
    reference_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> dict[str, float]:
    """Compare one trajectory with the finest-dt reference trajectory."""
    reference_time, _ = _unique_time_series(
        reference_rows,
        COMPARISON_FIELDS[0],
    )
    metrics: dict[str, float] = {}

    for field in COMPARISON_FIELDS:
        reference_field_time, reference_values = _unique_time_series(
            reference_rows,
            field,
        )
        candidate_time, candidate_values = _unique_time_series(
            candidate_rows,
            field,
        )
        if not np.allclose(reference_time, reference_field_time):
            raise ValueError("reference trajectory fields use different grids")
        interpolated = np.interp(
            reference_time,
            candidate_time,
            candidate_values,
        )
        difference = interpolated - reference_values
        metrics[f"{field}_max_abs"] = float(np.max(np.abs(difference)))
        metrics[f"{field}_rms"] = float(
            np.sqrt(np.mean(difference * difference))
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dt-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_DT_VALUES_S),
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--sample-every", type=float, default=1.0)
    parser.add_argument("--stop-time", type=float, default=480.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v11-reference"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v11-reference-convergence.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v11-reference-convergence.csv"),
    )
    args = parser.parse_args()

    if not args.dt_values or any(value <= 0.0 for value in args.dt_values):
        parser.error("--dt-values must contain positive values")
    if len(set(args.dt_values)) != len(args.dt_values):
        parser.error("--dt-values must be unique")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.sample_every <= 0.0:
        parser.error("--sample-every must be positive")
    if args.stop_time <= 0.0:
        parser.error("--stop-time must be positive")

    dt_values = sorted(args.dt_values, reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = min(args.jobs, len(dt_values))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_one_case,
                dt_s,
                args.sample_every,
                str(args.output_dir),
                args.stop_time,
            )
            for dt_s in dt_values
        ]
        summaries = [future.result() for future in futures]

    summaries.sort(key=lambda item: float(item["dt_s"]), reverse=True)
    successful = [item for item in summaries if item["status"] == "ok"]
    if not successful:
        raise RuntimeError("all V11 reference time-step runs failed")
    finest = min(successful, key=lambda item: float(item["dt_s"]))
    reference_rows = _load_csv(Path(str(finest["csv_path"])))

    metric_names = [
        f"{field}_{kind}"
        for field in COMPARISON_FIELDS
        for kind in ("max_abs", "rms")
    ]
    convergence_rows: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["status"] == "ok":
            candidate_rows = _load_csv(Path(str(summary["csv_path"])))
            metrics: dict[str, Any] = convergence_metrics(
                reference_rows=reference_rows,
                candidate_rows=candidate_rows,
            )
        else:
            metrics = {name: None for name in metric_names}
        convergence_rows.append({**summary, **metrics})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(convergence_rows[0]),
        )
        writer.writeheader()
        writer.writerows(convergence_rows)

    output = {
        "schema_version": 1,
        "model": "v11-reference-dynamic-convergence",
        "parallel_workers": workers,
        "dt_values_s": dt_values,
        "finest_dt_s": float(finest["dt_s"]),
        "sample_every_s": args.sample_every,
        "stop_time_s": args.stop_time,
        "comparison_fields": list(COMPARISON_FIELDS),
        "runs": convergence_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    for row in convergence_rows:
        if row["status"] == "ok":
            print(
                f"dt={float(row['dt_s']):.6g} s "
                f"Vmax={float(row['cell_voltage_v_max_abs']):.3e} V "
                f"Tmax={float(row['stack_temperature_k_max_abs']):.3e} K "
                f"lambda_max="
                f"{float(row['membrane_mean_water_content_max_abs']):.3e} "
                f"purges={int(row['purge_count'])}",
                flush=True,
            )
        else:
            print(
                f"dt={float(row['dt_s']):.6g} s FAILED: {row['error']}",
                flush=True,
            )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
