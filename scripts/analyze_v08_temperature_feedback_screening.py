"""Screen first-order temperature feedback on existing V0.7 closures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.anode import water_saturation_pressure_pa
from pemfc_dedalus.gas_permeability import (
    arrhenius_permeability,
    barrer_to_si_permeability,
)
from pemfc_dedalus.membrane import membrane_water_diffusivity_motupally
from pemfc_dedalus.parameters import CathodeParameters
from scripts.run_v07_n2_catalano_calibrated import (
    DRY_N2_ACTIVATION_ENERGY_J_MOL,
    DRY_REFERENCE_PERMEABILITY_BARRER,
    REFERENCE_TEMPERATURE_K,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ratio(value: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("reference must be non-zero")
    return value / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/quick-v08-lumped-thermal-dynamics.csv"),
    )
    parser.add_argument(
        "--representative-lambda",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v08-temperature-feedback-screening.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v08-temperature-feedback-screening.csv"),
    )
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if not rows:
        raise ValueError("temperature trajectory is empty")

    p = CathodeParameters()
    baseline_temperature_k = p.stack_temperature
    gas_constant = p.gas_constant
    dry_reference_si = barrer_to_si_permeability(
        DRY_REFERENCE_PERMEABILITY_BARRER
    )

    baseline_diffusivity = float(
        membrane_water_diffusivity_motupally(
            args.representative_lambda,
            baseline_temperature_k,
        ).item()
    )
    baseline_permeability = arrhenius_permeability(
        dry_reference_si,
        baseline_temperature_k,
        REFERENCE_TEMPERATURE_K,
        DRY_N2_ACTIVATION_ENERGY_J_MOL,
        gas_constant,
    )
    baseline_saturation_pressure = water_saturation_pressure_pa(
        baseline_temperature_k
    )
    baseline_inverse_temperature = 1.0 / baseline_temperature_k

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        temperature_k = float(row["stack_temperature_c"]) + 273.15
        diffusivity = float(
            membrane_water_diffusivity_motupally(
                args.representative_lambda,
                temperature_k,
            ).item()
        )
        permeability = arrhenius_permeability(
            dry_reference_si,
            temperature_k,
            REFERENCE_TEMPERATURE_K,
            DRY_N2_ACTIVATION_ENERGY_J_MOL,
            gas_constant,
        )
        saturation_pressure = water_saturation_pressure_pa(temperature_k)
        inverse_temperature = 1.0 / temperature_k

        output_rows.append(
            {
                "time_s": float(row["time_s"]),
                "stack_temperature_c": float(row["stack_temperature_c"]),
                "motupally_diffusivity_m2_s": diffusivity,
                "motupally_ratio_to_v07": ratio(
                    diffusivity,
                    baseline_diffusivity,
                ),
                "dry_n2_permeability_si": permeability,
                "dry_n2_permeability_ratio_to_v07": ratio(
                    permeability,
                    baseline_permeability,
                ),
                "water_saturation_pressure_pa": saturation_pressure,
                "water_saturation_pressure_ratio_to_v07": ratio(
                    saturation_pressure,
                    baseline_saturation_pressure,
                ),
                "inverse_temperature_ratio_to_v07": ratio(
                    inverse_temperature,
                    baseline_inverse_temperature,
                ),
            }
        )

    def extrema(key: str) -> tuple[float, float]:
        values = [float(row[key]) for row in output_rows]
        return min(values), max(values)

    d_min, d_max = extrema("motupally_ratio_to_v07")
    p_min, p_max = extrema("dry_n2_permeability_ratio_to_v07")
    psat_min, psat_max = extrema(
        "water_saturation_pressure_ratio_to_v07"
    )
    invt_min, invt_max = extrema("inverse_temperature_ratio_to_v07")

    summary = {
        "schema_version": 1,
        "model": "v08-temperature-feedback-screening",
        "baseline_temperature_k": baseline_temperature_k,
        "baseline_temperature_c": baseline_temperature_k - 273.15,
        "representative_lambda": args.representative_lambda,
        "motupally_ratio_min": d_min,
        "motupally_ratio_max": d_max,
        "dry_n2_permeability_ratio_min": p_min,
        "dry_n2_permeability_ratio_max": p_max,
        "water_saturation_pressure_ratio_min": psat_min,
        "water_saturation_pressure_ratio_max": psat_max,
        "inverse_temperature_ratio_min": invt_min,
        "inverse_temperature_ratio_max": invt_max,
        "final": output_rows[-1],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        "Motupally ratio="
        f"{d_min:.3f}..{d_max:.3f}, "
        "N2 permeability ratio="
        f"{p_min:.3f}..{p_max:.3f}, "
        "p_sat ratio="
        f"{psat_min:.3f}..{psat_max:.3f}, "
        "1/T ratio="
        f"{invt_min:.3f}..{invt_max:.3f}",
        flush=True,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
