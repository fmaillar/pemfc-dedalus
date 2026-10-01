"""Run a configurable V11 time-domain PEMFC simulation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pemfc_dedalus.v11_purge import V11PurgeClock
from pemfc_dedalus.v11_runner import (
    V11ControlSegment,
    V11RunnerInputs,
    run_v11_dynamic,
    trajectory_to_rows,
)
from pemfc_dedalus.v11_system import V11DynamicState


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("V11 dynamic config root must be a JSON object")
    return config


def _state_from_config(config: dict[str, Any]) -> V11DynamicState:
    values = config["initial_state"]
    return V11DynamicState(
        stack_temperature_k=float(values["stack_temperature_k"]),
        membrane_mean_water_content=float(
            values["membrane_mean_water_content"]
        ),
        anode_hydrogen_mol=float(values["anode_hydrogen_mol"]),
        anode_nitrogen_mol=float(values["anode_nitrogen_mol"]),
        anode_water_vapour_mol=float(values["anode_water_vapour_mol"]),
        cathode_oxygen_mol=float(values["cathode_oxygen_mol"]),
        cathode_nitrogen_mol=float(values["cathode_nitrogen_mol"]),
        cathode_total_water_mol=float(values["cathode_total_water_mol"]),
    )


def _controls_from_config(
    config: dict[str, Any],
) -> tuple[V11ControlSegment, ...]:
    return tuple(
        V11ControlSegment(
            start_time_s=float(item["start_time_s"]),
            current_a=float(item["current_a"]),
            stack_air_flow_slpm=float(item["stack_air_flow_slpm"]),
        )
        for item in config["controls"]
    )


def _inputs_from_config(config: dict[str, Any]) -> V11RunnerInputs:
    values = config["inputs"]
    return V11RunnerInputs(
        inlet_air_temperature_k=float(values["inlet_air_temperature_k"]),
        cathode_total_pressure_pa=float(
            values["cathode_total_pressure_pa"]
        ),
        inlet_oxygen_mole_fraction=float(
            values["inlet_oxygen_mole_fraction"]
        ),
        inlet_water_mole_fraction=float(
            values["inlet_water_mole_fraction"]
        ),
        cathode_platinum_loading_mg_cm2_geo=float(
            values["cathode_platinum_loading_mg_cm2_geo"]
        ),
        cathode_ecsa_m2_pt_g_pt=float(
            values["cathode_ecsa_m2_pt_g_pt"]
        ),
        additional_resolved_loss_v=float(
            values.get("additional_resolved_loss_v", 0.0)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/v11-dynamic.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/v11-dynamic.json"),
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    initial_clock = V11PurgeClock(
        charge_since_purge_as=float(
            config.get("initial_charge_since_purge_as", 0.0)
        )
    )
    trajectory = run_v11_dynamic(
        initial_state=_state_from_config(config),
        controls=_controls_from_config(config),
        inputs=_inputs_from_config(config),
        stop_time_s=float(config["stop_time_s"]),
        dt_s=float(config["dt_s"]),
        automatic_purge=bool(config.get("automatic_purge", True)),
        manual_purge_times_s=tuple(
            float(value)
            for value in config.get("manual_purge_times_s", [])
        ),
        initial_purge_clock=initial_clock,
        sample_every_s=(
            None
            if config.get("sample_every_s") is None
            else float(config["sample_every_s"])
        ),
    )
    rows = trajectory_to_rows(trajectory)
    if not rows:
        raise RuntimeError("V11 dynamic runner returned no trajectory rows")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    final = rows[-1]
    output = {
        "schema_version": 1,
        "model": "v11-dynamic-runner",
        "config": str(args.config),
        "rows": len(rows),
        "stop_time_s": float(final["time_s"]),
        "purge_count": int(final["purge_count"]),
        "final_cell_voltage_v": float(final["cell_voltage_v"]),
        "final_stack_temperature_k": float(final["stack_temperature_k"]),
        "final_membrane_mean_water_content": float(
            final["membrane_mean_water_content"]
        ),
        "output_csv": str(args.output_csv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    print(
        f"rows={len(rows)} "
        f"purges={output['purge_count']} "
        f"Vfinal={output['final_cell_voltage_v']:.6f} V "
        f"Tfinal={output['final_stack_temperature_k']:.3f} K",
        flush=True,
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
