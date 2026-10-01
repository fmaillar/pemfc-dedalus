"""Analyze convergence of the V0.8 Motupally temperature lookup axis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    "mean_n2_crossover_flux_mol_m2_s",
    "max_nitrogen_mole_fraction",
    "cumulative_n2_crossover_mol",
    "cumulative_n2_purged_mol",
    "mean_purge_period_s",
)


def relative_change(value: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("reference must be non-zero")
    return value / reference - 1.0


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return data["summaries"]["full_temperature_lookup"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        nargs="+",
        type=int,
        default=[3, 5, 7, 9],
    )
    parser.add_argument(
        "--input-pattern",
        default="results/v08-motupally-temperature-convergence-{points}.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v08-motupally-temperature-convergence.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v08-motupally-temperature-convergence.csv"),
    )
    args = parser.parse_args()

    if len(args.points) < 2:
        parser.error("at least two temperature resolutions are required")
    if any(points < 2 for points in args.points):
        parser.error("all temperature resolutions must be >= 2")

    summaries = {
        points: load_summary(
            Path(args.input_pattern.format(points=points))
        )
        for points in args.points
    }
    reference_points = max(args.points)
    reference = summaries[reference_points]

    rows: list[dict[str, Any]] = []
    for points in sorted(args.points):
        summary = summaries[points]
        for metric in METRICS:
            value = float(summary[metric])
            reference_value = float(reference[metric])
            rows.append(
                {
                    "temperature_points": points,
                    "reference_temperature_points": reference_points,
                    "metric": metric,
                    "value": value,
                    "reference_value": reference_value,
                    "relative_change": relative_change(
                        value,
                        reference_value,
                    ),
                    "absolute_relative_change": abs(
                        relative_change(value, reference_value)
                    ),
                }
            )

    non_reference = [
        row
        for row in rows
        if int(row["temperature_points"]) != reference_points
    ]
    max_abs_change = max(
        float(row["absolute_relative_change"])
        for row in non_reference
    )

    by_points: dict[int, float] = {}
    for points in sorted(args.points):
        point_rows = [
            row for row in rows if int(row["temperature_points"]) == points
        ]
        by_points[points] = max(
            float(row["absolute_relative_change"])
            for row in point_rows
        )

    output = {
        "schema_version": 1,
        "model": "v08-motupally-temperature-convergence",
        "temperature_points": sorted(args.points),
        "reference_temperature_points": reference_points,
        "max_absolute_relative_change_non_reference": max_abs_change,
        "max_absolute_relative_change_by_points": by_points,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for points in sorted(args.points):
        print(
            f"N_T={points}: max relative difference vs "
            f"N_T={reference_points} = "
            f"{100.0 * by_points[points]:.4f}%",
            flush=True,
        )

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
