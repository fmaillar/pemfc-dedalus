import numpy as np
import pytest

from scripts.run_v07_anode_purge import simulate_purge_regime


def test_charge_triggered_purge_cadence_and_balances():
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, 0.0, -2.0e-6]),
        np.asarray([0.0020, 0.0022, 0.0024]),
    )
    rows, summary = simulate_purge_regime(
        "nominal",
        closure,
        initial_rh=0.0,
        stop_time_s=12.0,
        dt_s=0.1,
        write_every=10,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
        faraday_c_mol=96485.33212,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
        target_total_pressure_pa=137325.0,
        purge_interval_as=10.0,
        purge_clock_current_a=2.0,
        purge_exchange_volume_m3=20.0e-6,
    )

    assert rows
    assert summary["purge_count"] == 2
    assert summary["expected_purge_period_s"] == pytest.approx(5.0)
    assert summary["first_purge_time_s"] == pytest.approx(5.0, abs=0.11)
    assert summary["mean_purge_period_s"] == pytest.approx(5.0, abs=0.11)
    assert abs(summary["h2_balance_error_mol"]) < 1.0e-14
    assert abs(summary["water_balance_error_mol"]) < 1.0e-14
    assert summary["cumulative_h2_purged_mol"] > 0.0
    assert summary["cumulative_water_purged_mol"] > 0.0


def test_purge_fraction_is_between_zero_and_one():
    closure = (
        np.asarray([0.0, 1.0]),
        np.asarray([1.0e-6, -1.0e-6]),
        np.asarray([0.0020, 0.0021]),
    )
    _, summary = simulate_purge_regime(
        "nominal",
        closure,
        initial_rh=0.0,
        stop_time_s=6.0,
        dt_s=0.1,
        write_every=10,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
        faraday_c_mol=96485.33212,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
        target_total_pressure_pa=137325.0,
        purge_interval_as=10.0,
        purge_clock_current_a=2.0,
        purge_exchange_volume_m3=20.0e-6,
    )

    assert 0.0 < summary["purge_fraction"] < 1.0
