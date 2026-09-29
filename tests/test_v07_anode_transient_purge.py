import numpy as np
import pytest

from scripts.run_v07_anode_transient_purge import (
    simulate_transient_purge_regime,
)


def test_transient_purge_resolves_finite_duration_and_balances():
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, 0.0, -2.0e-6]),
        np.asarray([0.0020, 0.0022, 0.0024]),
    )
    rows, summary = simulate_transient_purge_regime(
        "nominal",
        closure,
        initial_rh=0.0,
        stop_time_s=12.0,
        dt_s=0.01,
        write_every=100,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
        faraday_c_mol=96485.33212,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
        target_total_pressure_pa=137325.0,
        ambient_pressure_pa=101325.0,
        purge_interval_as=10.0,
        purge_clock_current_a=2.0,
        purge_duration_s=0.2,
        purge_reference_flow_slpm=2.4,
    )

    assert rows
    assert summary["purge_count"] == 2
    assert summary["expected_purge_period_s"] == pytest.approx(5.0)
    assert summary["first_purge_time_s"] == pytest.approx(5.0, abs=0.011)
    assert summary["mean_purge_period_s"] == pytest.approx(5.0, abs=0.011)
    assert summary["min_total_pressure_pa"] > 101325.0
    assert summary["min_total_pressure_pa"] < 137325.0
    assert abs(summary["h2_balance_error_mol"]) < 1.0e-13
    assert abs(summary["water_balance_error_mol"]) < 1.0e-13

    first_event = summary["purge_events"][0]
    assert first_event["end_time_s"] - first_event["start_time_s"] == pytest.approx(
        0.2,
        abs=0.011,
    )
    assert first_event["h2_purged_mol"] > 0.0
    assert first_event["water_purged_mol"] > 0.0
    assert first_event["h2_refill_mol"] > 0.0
    assert first_event["max_outflow_mol_s"] > 0.0



def test_dynamic_cell_current_drives_purge_clock():
    closure = (
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0020, 0.0020]),
    )
    _, summary = simulate_transient_purge_regime(
        "nominal",
        closure,
        initial_rh=0.0,
        stop_time_s=12.0,
        dt_s=0.01,
        write_every=100,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
        faraday_c_mol=96485.33212,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
        target_total_pressure_pa=137325.0,
        ambient_pressure_pa=101325.0,
        purge_interval_as=10.0,
        purge_clock_current_a=None,
        purge_duration_s=0.2,
        purge_reference_flow_slpm=2.4,
        current_scale_factor=1000.0,
    )

    assert summary["purge_clock_mode"] == "dynamic_cell_current"
    assert summary["purge_clock_current_a"] is None
    assert summary["expected_purge_period_s"] is None
    assert summary["purge_count"] == 2
    assert summary["first_purge_time_s"] == pytest.approx(5.0, abs=0.011)
    assert summary["mean_purge_period_s"] == pytest.approx(5.0, abs=0.011)
