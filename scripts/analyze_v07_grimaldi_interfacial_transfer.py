"""Diagnose Grimaldi interfacial water coefficients at stack temperature."""

from __future__ import annotations

import csv
from pathlib import Path

from pemfc_dedalus.membrane import (
    nafion_water_interfacial_transfer_coefficient_grimaldi,
)
from pemfc_dedalus.parameters import CathodeParameters

REPRESENTATIVE_LAMBDA = {
    "dry_high_load": (2.95, 3.49),
    "nominal": (3.24, 3.49),
    "wet_low_load": (3.49, 5.37),
}


def main() -> None:
    p = CathodeParameters()
    baseline = p.anode_water_transfer_coefficient
    rows: list[dict[str, float | str]] = []

    for regime, (lambda_min, lambda_max) in REPRESENTATIVE_LAMBDA.items():
        for state, water_content in (
            ("lambda_min", lambda_min),
            ("lambda_max", lambda_max),
        ):
            coefficient = float(
                nafion_water_interfacial_transfer_coefficient_grimaldi(
                    water_content,
                    p.stack_temperature,
                    gas_constant_j_mol_k=p.gas_constant,
                ).item()
            )
            rows.append(
                {
                    "regime": regime,
                    "state": state,
                    "lambda": water_content,
                    "stack_temperature_k": p.stack_temperature,
                    "grimaldi_coefficient_m_s": coefficient,
                    "current_baseline_k_a_m_s": baseline,
                    "grimaldi_to_current_ratio": coefficient / baseline,
                }
            )

    output = Path(
        "results/v07-grimaldi-interfacial-transfer-diagnostic.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['regime']} {row['state']} "
            f"lambda={row['lambda']:.2f}: "
            f"k_g={row['grimaldi_coefficient_m_s']:.3e} m/s, "
            f"ratio={row['grimaldi_to_current_ratio']:.2f}",
            flush=True,
        )

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
