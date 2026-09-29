import numpy as np

from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime


def test_nitrogen_driver_conserves_all_species():
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, 0.0, -2.0e-6]),
        np.asarray([0.0020, 0.0022, 0.0024]),
    )
    rows, summary = simulate_nitrogen_regime(
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
        active_area_m2=0.02,
        fixed_charge_mol_m3=1800.0,
        target_total_pressure_pa=137325.0,
        ambient_pressure_pa=101325.0,
        purge_interval_as=10.0,
        purge_duration_s=0.5,
        purge_reference_flow_slpm=2.4,
        current_scale_factor=1000.0,
        n2_crossover_flux_mol_m2_s=1.0e-5,
    )

    assert rows
    assert summary["purge_count"] == 2
    assert summary["cumulative_n2_crossover_mol"] > 0.0
    assert summary["cumulative_n2_purged_mol"] > 0.0
    assert summary["max_nitrogen_mole_fraction"] > 0.0
    assert summary["min_hydrogen_mole_fraction"] < 1.0
    assert abs(summary["h2_balance_error_mol"]) < 1.0e-12
    assert abs(summary["n2_balance_error_mol"]) < 1.0e-12
    assert abs(summary["water_balance_error_mol"]) < 1.0e-12
    assert summary["min_total_pressure_pa"] > 101325.0


def test_zero_crossover_keeps_nitrogen_inventory_zero():
    closure = (
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0020, 0.0020]),
    )
    _, summary = simulate_nitrogen_regime(
        "nominal",
        closure,
        initial_rh=0.0,
        stop_time_s=6.0,
        dt_s=0.01,
        write_every=100,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
        faraday_c_mol=96485.33212,
        active_area_m2=0.02,
        fixed_charge_mol_m3=1800.0,
        target_total_pressure_pa=137325.0,
        ambient_pressure_pa=101325.0,
        purge_interval_as=10.0,
        purge_duration_s=0.5,
        purge_reference_flow_slpm=2.4,
        current_scale_factor=1000.0,
        n2_crossover_flux_mol_m2_s=0.0,
    )

    assert summary["final_nitrogen_mol"] == 0.0
    assert summary["cumulative_n2_crossover_mol"] == 0.0
    assert summary["cumulative_n2_purged_mol"] == 0.0
    assert summary["n2_balance_error_mol"] == 0.0
