"""Diagnose Ge et al. (2005) Nafion/gas interfacial water coefficients."""

from __future__ import annotations

import csv
from pathlib import Path

from pemfc_dedalus.gas_permeability import (
    membrane_water_volume_fraction_from_partial_molar_volume,
)
from pemfc_dedalus.membrane import (
    nafion_water_interfacial_transfer_coefficient_ge,
)
from pemfc_dedalus.parameters import CathodeParameters

REPRESENTATIVE_LAMBDA = {
    "dry_high_load": (2.95, 3.49),
    "nominal": (3.24, 3.49),
    "wet_low_load": (3.49, 5.37),
}


def main() -> None:
    p = CathodeParameters()
    rows: list[dict[str, float | str]] = []

    for regime, (lambda_min, lambda_max) in REPRESENTATIVE_LAMBDA.items():
        for label, water_content in (
            ("lambda_min", lambda_min),
            ("lambda_max", lambda_max),
        ):
            fraction = membrane_water_volume_fraction_from_partial_molar_volume(
                water_content,
                membrane_equivalent_weight_kg_mol=p.membrane_equivalent_weight,
                membrane_dry_density_kg_m3=p.membrane_dry_density,
                water_partial_molar_volume_m3_mol=17.0e-6,
            )
            absorption = float(
                nafion_water_interfacial_transfer_coefficient_ge(
                    fraction,
                    mode="absorption",
                ).item()
            )
            desorption = float(
                nafion_water_interfacial_transfer_coefficient_ge(
                    fraction,
                    mode="desorption",
                ).item()
            )
            rows.append(
                {
                    "regime": regime,
                    "state": label,
                    "lambda": water_content,
                    "water_volume_fraction": fraction,
                    "ge_absorption_coefficient_m_s_353k": absorption,
                    "ge_desorption_coefficient_m_s_353k": desorption,
                    "current_baseline_k_a_m_s_313k": (
                        p.anode_water_transfer_coefficient
                    ),
                    "absorption_to_current_ratio": (
                        absorption / p.anode_water_transfer_coefficient
                    ),
                    "desorption_to_current_ratio": (
                        desorption / p.anode_water_transfer_coefficient
                    ),
                }
            )

    output = Path("results/v07-ge-interfacial-transfer-diagnostic.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['regime']} {row['state']} lambda={row['lambda']:.2f}: "
            f"fv={row['water_volume_fraction']:.3f}, "
            f"k_abs={row['ge_absorption_coefficient_m_s_353k']:.3e}, "
            f"k_des={row['ge_desorption_coefficient_m_s_353k']:.3e}",
            flush=True,
        )

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
