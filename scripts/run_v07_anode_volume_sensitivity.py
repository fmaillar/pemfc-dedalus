"""V0.7 sensitivity of anode humidification dynamics to gas volume."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.membrane import membrane_fixed_charge_concentration
from pemfc_dedalus.parameters import CathodeParameters
from scripts.run_v07_anode_volume import (
    REGIMES,
    load_flux_closure,
    simulate_regime,
)

DEFAULT_VOLUMES_M3 = (5.0e-6, 10.0e-6, 20.0e-6, 40.0e-6, 80.0e-6)


def first_time_to_fraction(
    rows: list[dict[str, Any]],
    equilibrium_rh: float,
    fraction: float,
) -> float | None:
    """Return first sampled time reaching a fraction of the equilibrium RH."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    target = fraction * equilibrium_rh
    for row in rows:
        if float(row["anode_relative_humidity"]) >= target:
            return float(row["time_s"])
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty V0.7 volume-sensitivity table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure-csv",
        type=Path,
        default=Path("results/v06-rha-sensitivity.csv"),
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=list(REGIMES),
        default=list(REGIMES),
    )
    parser.add_argument(
        "--volumes-ml",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0, 40.0, 80.0],
    )
    parser.add_argument("--initial-rh", type=float, default=0.0)
    parser.add_argument(
        "--horizon-per-ml",
        type=float,
        default=750.0,
        help="simulation horizon [s] per mL of anode gas volume",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--write-every", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v07-anode-volume-sensitivity.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v07-anode-volume-sensitivity.csv"),
    )
    args = parser.parse_args()

    if any(volume_ml <= 0.0 for volume_ml in args.volumes_ml):
        parser.error("--volumes-ml must all be positive")
    if not 0.0 <= args.initial_rh <= 1.0:
        parser.error("--initial-rh must be in [0, 1]")
    if args.horizon_per_ml <= 0.0:
        parser.error("--horizon-per-ml must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.write_every < 1:
        parser.error("--write-every must be >= 1")

    p = CathodeParameters()
    closure = load_flux_closure(args.closure_csv)
    missing = [regime for regime in args.regimes if regime not in closure]
    if missing:
        parser.error(f"missing closure data for regimes: {', '.join(missing)}")

    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    membrane_area = p.length_x * p.length_y

    summaries: list[dict[str, Any]] = []

    for regime in args.regimes:
        for volume_ml in args.volumes_ml:
            volume_m3 = volume_ml * 1.0e-6
            stop_time_s = args.horizon_per_ml * volume_ml
            rows, summary = simulate_regime(
                regime,
                closure[regime],
                initial_rh=args.initial_rh,
                stop_time_s=stop_time_s,
                dt_s=args.dt,
                volume_m3=volume_m3,
                temperature_k=p.stack_temperature,
                gas_constant_j_mol_k=p.gas_constant,
                membrane_area_m2=membrane_area,
                fixed_charge_mol_m3=fixed_charge,
                write_every=args.write_every,
            )

            equilibrium = summary["static_zero_flux_relative_humidity"]
            if equilibrium is None:
                raise RuntimeError(f"{regime}: no static zero-flux RH in closure")

            record = {
                "regime": regime,
                "anode_gas_volume_ml": volume_ml,
                "anode_gas_volume_m3": volume_m3,
                "stop_time_s": stop_time_s,
                "equilibrium_relative_humidity": equilibrium,
                "final_relative_humidity": summary["final_relative_humidity"],
                "absolute_rh_error_to_equilibrium": summary[
                    "absolute_rh_error_to_static_zero_flux"
                ],
                "final_water_flux_lambda_m_s": summary[
                    "final_water_flux_lambda_m_s"
                ],
                "final_liquid_water_mol": summary["final_liquid_water_mol"],
                "t50_s": first_time_to_fraction(rows, equilibrium, 0.50),
                "t90_s": first_time_to_fraction(rows, equilibrium, 0.90),
                "t95_s": first_time_to_fraction(rows, equilibrium, 0.95),
                "t99_s": first_time_to_fraction(rows, equilibrium, 0.99),
            }
            summaries.append(record)
            print(
                f"{regime} V={volume_ml:g} mL: "
                f"t90={record['t90_s']} s t99={record['t99_s']} s "
                f"RH_final={record['final_relative_humidity']:.6f}",
                flush=True,
            )

    output = {
        "schema_version": 1,
        "model": "v07-anode-volume-sensitivity",
        "closure_source": str(args.closure_csv),
        "initial_relative_humidity": args.initial_rh,
        "horizon_per_ml_s_ml": args.horizon_per_ml,
        "dt_s": args.dt,
        "write_every": args.write_every,
        "volumes_ml": args.volumes_ml,
        "regimes": args.regimes,
        "cases": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_csv, summaries)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
