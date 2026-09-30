"""Compare constant and Motupally Nafion water diffusivity on V0.7 profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from pemfc_dedalus.membrane import (
    membrane_water_diffusivity_motupally,
    steady_membrane_water_profile,
)
from pemfc_dedalus.parameters import CathodeParameters

REFERENCE_DIFFUSIVITY_M2_S = 2.0e-10


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty Motupally diffusivity diagnostics")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "results/v07-n2-catalano-membrane-profile-diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/v07-n2-motupally-diffusivity-diagnostics.csv"
        ),
    )
    args = parser.parse_args()

    p = CathodeParameters()
    input_rows = list(csv.DictReader(args.input_csv.open(newline="")))
    rows: list[dict[str, Any]] = []

    for source in input_rows:
        diffusivity = float(source["membrane_water_diffusivity_m2_s"])
        peclet = float(source["water_peclet"])
        lambda_anode = float(source["lambda_anode_profile"])
        lambda_cathode = float(source["lambda_cathode"])

        drag_velocity = peclet * diffusivity / p.membrane_thickness
        z = np.linspace(
            0.0,
            p.membrane_thickness,
            129,
        )
        lambda_profile = steady_membrane_water_profile(
            z,
            lambda_anode=lambda_anode,
            lambda_cathode=lambda_cathode,
            diffusivity_m2_s=diffusivity,
            drag_velocity_m_s=drag_velocity,
        )
        motupally = membrane_water_diffusivity_motupally(
            lambda_profile,
            p.stack_temperature,
        )

        rows.append(
            {
                "regime": source["regime"],
                "hydrogen_feedback_exponent": float(
                    source["hydrogen_feedback_exponent"]
                ),
                "source_diffusivity_m2_s": diffusivity,
                "anode_transfer_coefficient_m_s": float(
                    source["anode_transfer_coefficient_m_s"]
                ),
                "lambda_min": float(np.min(lambda_profile)),
                "lambda_mean": float(np.mean(lambda_profile)),
                "lambda_max": float(np.max(lambda_profile)),
                "motupally_diffusivity_min_m2_s": float(np.min(motupally)),
                "motupally_diffusivity_mean_m2_s": float(np.mean(motupally)),
                "motupally_diffusivity_max_m2_s": float(np.max(motupally)),
                "motupally_mean_to_reference_ratio": float(
                    np.mean(motupally) / REFERENCE_DIFFUSIVITY_M2_S
                ),
                "motupally_min_to_reference_ratio": float(
                    np.min(motupally) / REFERENCE_DIFFUSIVITY_M2_S
                ),
                "motupally_max_to_reference_ratio": float(
                    np.max(motupally) / REFERENCE_DIFFUSIVITY_M2_S
                ),
                "profile_crosses_lambda_three": bool(
                    np.min(lambda_profile) <= 3.0 < np.max(lambda_profile)
                ),
                "fraction_profile_lambda_le_three": float(
                    np.mean(lambda_profile <= 3.0)
                ),
                "source_mean_n2_crossover_flux_mol_m2_s": float(
                    source["source_mean_n2_crossover_flux_mol_m2_s"]
                ),
                "source_max_nitrogen_mole_fraction": float(
                    source["source_max_nitrogen_mole_fraction"]
                ),
            }
        )

    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
