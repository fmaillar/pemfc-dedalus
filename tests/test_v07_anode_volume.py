import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.run_v07_anode_volume import (
    interpolate_lambda_flux,
    load_flux_closure,
    simulate_regime,
    zero_flux_relative_humidity,
)


def test_load_flux_closure_groups_and_sorts(tmp_path: Path):
    path = tmp_path / "closure.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "regime",
                "anode_relative_humidity",
                "final_anode_water_removal_flux_lambda_m_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "regime": "nominal",
                "anode_relative_humidity": 0.5,
                "final_anode_water_removal_flux_lambda_m_s": -1.0e-6,
            }
        )
        writer.writerow(
            {
                "regime": "nominal",
                "anode_relative_humidity": 0.0,
                "final_anode_water_removal_flux_lambda_m_s": 2.0e-6,
            }
        )

    rh, flux = load_flux_closure(path)["nominal"]
    assert np.allclose(rh, [0.0, 0.5])
    assert np.allclose(flux, [2.0e-6, -1.0e-6])


def test_flux_interpolation_and_zero_crossing_are_consistent():
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, -1.0e-6, -2.0e-6]),
    )
    assert interpolate_lambda_flux(0.25, closure) == pytest.approx(0.5e-6)
    assert zero_flux_relative_humidity(closure) == pytest.approx(1.0 / 3.0)


def test_dynamic_anode_moves_toward_static_zero_flux_point():
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, -1.0e-6, -2.0e-6]),
    )
    rows, summary = simulate_regime(
        "nominal",
        closure,
        initial_rh=0.0,
        stop_time_s=10000.0,
        dt_s=1.0,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
        write_every=100,
    )

    assert rows
    assert summary["final_relative_humidity"] > 0.0
    assert summary["static_zero_flux_relative_humidity"] == pytest.approx(1.0 / 3.0)
    assert summary["absolute_rh_error_to_static_zero_flux"] < 0.05
    assert abs(summary["final_water_flux_lambda_m_s"]) < 5.0e-7
