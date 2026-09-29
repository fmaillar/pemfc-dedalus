import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.run_v07_anode_h2 import (
    interpolate_flux_and_current,
    load_h2_closure,
    simulate_h2_regime,
)


def test_h2_closure_loads_flux_and_current(tmp_path: Path):
    path = tmp_path / "closure.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "regime",
                "anode_relative_humidity",
                "final_anode_water_removal_flux_lambda_m_s",
                "final_current_a",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "regime": "nominal",
                "anode_relative_humidity": 0.5,
                "final_anode_water_removal_flux_lambda_m_s": -1.0e-6,
                "final_current_a": 0.0022,
            }
        )
        writer.writerow(
            {
                "regime": "nominal",
                "anode_relative_humidity": 0.0,
                "final_anode_water_removal_flux_lambda_m_s": 2.0e-6,
                "final_current_a": 0.0020,
            }
        )

    rh, flux, current = load_h2_closure(path)["nominal"]
    assert np.allclose(rh, [0.0, 0.5])
    assert np.allclose(flux, [2.0e-6, -1.0e-6])
    assert np.allclose(current, [0.0020, 0.0022])


def test_h2_closure_interpolates_flux_and_current_together():
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, 0.0, -2.0e-6]),
        np.asarray([0.0020, 0.0022, 0.0024]),
    )
    flux, current = interpolate_flux_and_current(0.25, closure)
    assert flux == pytest.approx(1.0e-6)
    assert current == pytest.approx(0.0021)


def test_pressure_regulated_h2_simulation_preserves_positive_inventory():
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, 0.0, -2.0e-6]),
        np.asarray([0.0020, 0.0022, 0.0024]),
    )
    rows, summary = simulate_h2_regime(
        "nominal",
        closure,
        initial_rh=0.0,
        stop_time_s=100.0,
        dt_s=1.0,
        write_every=10,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
        faraday_c_mol=96485.33212,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
        target_total_pressure_pa=137325.0,
    )

    assert rows
    assert summary["final_hydrogen_mol"] > 0.0
    assert summary["cumulative_h2_consumed_mol"] > 0.0
    assert summary["cumulative_h2_inlet_mol"] >= 0.0
    assert summary["final_total_pressure_pa"] > 0.0
    assert summary["max_total_pressure_pa"] >= 137325.0
