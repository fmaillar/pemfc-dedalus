"""Compare 21x21 and 33x33 Motupally lookup quick campaigns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


KEYS = ("regime", "hydrogen_feedback_exponent")
METRICS = (
    "motupally_jmean_mol_m2_s",
    "motupally_xn2_max",
    "motupally_mean_cell_current_a",
)


def load_rows(path: Path) -> dict[tuple[str, float], dict[str, str]]:
    rows: dict[tuple[str, float], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["regime"], float(row["hydrogen_feedback_exponent"]))
            rows[key] = row
    return rows


def relative_change(new: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if new == 0.0 else float("inf")
    return new / reference - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coarse-csv",
        type=Path,
        default=Path(
            "results/quick-v07-n2-catalano-motupally-comparison.csv"
        ),
    )
    parser.add_argument(
        "--fine-csv",
        type=Path,
        default=Path(
            "results/quick-v07-n2-catalano-motupally-comparison-33x33.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/quick-v07-n2-catalano-motupally-lookup-convergence.csv"
        ),
    )
    args = parser.parse_args()

    coarse = load_rows(args.coarse_csv)
    fine = load_rows(args.fine_csv)
    if coarse.keys() != fine.keys():
        raise ValueError("coarse and fine campaigns do not contain the same cases")

    rows: list[dict[str, Any]] = []
    max_abs_change = 0.0

    for key in sorted(coarse, key=lambda item: (item[1], item[0])):
        coarse_row = coarse[key]
        fine_row = fine[key]
        row: dict[str, Any] = {
            "regime": key[0],
            "hydrogen_feedback_exponent": key[1],
        }
        for metric in METRICS:
            coarse_value = float(coarse_row[metric])
            fine_value = float(fine_row[metric])
            change = relative_change(fine_value, coarse_value)
            row[f"{metric}_21x21"] = coarse_value
            row[f"{metric}_33x33"] = fine_value
            row[f"{metric}_change_fraction"] = change
            max_abs_change = max(max_abs_change, abs(change))

        row["purge_count_21x21"] = int(coarse_row["motupally_purge_count"])
        row["purge_count_33x33"] = int(fine_row["motupally_purge_count"])
        coarse_period = coarse_row["motupally_mean_purge_period_s"]
        fine_period = fine_row["motupally_mean_purge_period_s"]
        row["purge_period_21x21_s"] = coarse_period
        row["purge_period_33x33_s"] = fine_period

        if coarse_period and fine_period:
            period_change = relative_change(
                float(fine_period),
                float(coarse_period),
            )
            row["purge_period_change_fraction"] = period_change
            max_abs_change = max(max_abs_change, abs(period_change))
        else:
            row["purge_period_change_fraction"] = ""

        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['regime']} gamma={row['hydrogen_feedback_exponent']:.1f}: "
            f"J change={100.0 * row['motupally_jmean_mol_m2_s_change_fraction']:+.3f}% "
            f"xN2 change={100.0 * row['motupally_xn2_max_change_fraction']:+.3f}%",
            flush=True,
        )
    print(f"max absolute relative change = {100.0 * max_abs_change:.3f}%")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
