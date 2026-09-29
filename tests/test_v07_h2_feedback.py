import numpy as np

from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime


def _run(feedback_exponent: float, n2_flux: float):
    closure = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([2.0e-6, 0.0, -2.0e-6]),
        np.asarray([0.0020, 0.0022, 0.0024]),
    )
    _, summary = simulate_nitrogen_regime(
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
        n2_crossover_flux_mol_m2_s=n2_flux,
        hydrogen_feedback_exponent=feedback_exponent,
    )
    return summary


def test_zero_feedback_exponent_preserves_base_current():
    summary = _run(0.0, 1.0e-5)
    assert summary["min_hydrogen_feedback_factor"] == 1.0


def test_hydrogen_feedback_reduces_current_with_nitrogen():
    disabled = _run(0.0, 1.0e-5)
    enabled = _run(1.0, 1.0e-5)

    assert enabled["min_hydrogen_feedback_factor"] < 1.0
    assert enabled["mean_cell_current_a"] < disabled["mean_cell_current_a"]


def test_zero_nitrogen_keeps_feedback_factor_unity():
    summary = _run(1.0, 0.0)
    assert abs(summary["min_hydrogen_feedback_factor"] - 1.0) < 1.0e-12
