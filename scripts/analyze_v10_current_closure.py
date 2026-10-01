"""Screen V1.0 electrochemical current closure against imposed stack current."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.parameters import CathodeParameters


INPUT = Path("results/quick-v09-quasi3d-current-map.json")
OUTPUT_JSON = Path("results/quick-v10-current-closure.json")
OUTPUT_CSV = Path("results/quick-v10-current-closure.csv")


def target_current_density_a_m2(
    *,
    current_a: float,
    reference_current_a: float,
    reference_current_density_a_m2: float,
) -> float:
    """Scale target current density from the validated nominal reference."""
    if current_a <= 0.0:
        raise ValueError("current_a must be positive")
    if reference_current_a <= 0.0:
        raise ValueError("reference_current_a must be positive")
    if reference_current_density_a_m2 <= 0.0:
        raise ValueError("reference_current_density_a_m2 must be positive")
    return reference_current_density_a_m2 * current_a / reference_current_a


def main() -> None:
    data = json.loads(INPUT.read_text())
    params = CathodeParameters()
    representative_area_m2 = params.length_x * params.length_y

    rows: list[dict[str, Any]] = []
    for source in data["rows"]:
        current_a = float(source["current_a"])
        reaction_current_a = float(source["total_reaction_current"])
        model_current_density = reaction_current_a / representative_area_m2
        target_current_density = target_current_density_a_m2(
            current_a=current_a,
            reference_current_a=params.stack_current_a,
            reference_current_density_a_m2=params.membrane_current_density,
        )
        relative_error = (
            model_current_density - target_current_density
        ) / target_current_density

        rows.append(
            {
                "current_a": current_a,
                "slice_xi": float(source["slice_xi"]),
                "cell_voltage_v": float(source["cell_voltage_v"]),
                "oxygen_feed_concentration_mol_m3": float(
                    source["oxygen_feed_concentration_mol_m3"]
                ),
                "reaction_current_a": reaction_current_a,
                "representative_area_m2": representative_area_m2,
                "model_current_density_a_m2": model_current_density,
                "target_current_density_a_m2": target_current_density,
                "relative_current_density_error": relative_error,
            }
        )

    max_abs_error = max(
        abs(float(row["relative_current_density_error"])) for row in rows
    )

    output = {
        "schema_version": 1,
        "model": "v10-electrochemical-current-closure-screening",
        "reference_current_a": params.stack_current_a,
        "reference_current_density_a_m2": params.membrane_current_density,
        "representative_area_m2": representative_area_m2,
        "max_abs_relative_current_density_error": max_abs_error,
        "rows": rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n")

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "V1.0 current-closure screening: "
        f"max_abs_error={100.0 * max_abs_error:.2f}%",
        flush=True,
    )
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
