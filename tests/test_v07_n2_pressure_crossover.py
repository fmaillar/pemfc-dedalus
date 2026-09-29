import numpy as np

from pemfc_dedalus.anode_nitrogen import (
    nitrogen_permeance_from_reference_flux,
)
from scripts.run_v07_anode_nitrogen import simulate_nitrogen_regime


def test_pressure_driven_crossover_declines_as_nitrogen_accumulates():
    closure = (
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0020, 0.0020]),
    )
    cathode_n2_pressure_pa = 80000.0
    reference_flux = 1.0e-5
    permeance = nitrogen_permeance_from_reference_flux(
        reference_flux,
        cathode_n2_pressure_pa,
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
        n2_crossover_flux_mol_m2_s=reference_flux,
        hydrogen_feedback_exponent=0.0,
        n2_crossover_permeance_mol_m2_s_pa=permeance,
        cathode_n2_partial_pressure_pa=cathode_n2_pressure_pa,
    )

    assert summary["n2_crossover_mode"] == "partial_pressure_driven"
    assert summary["max_n2_crossover_flux_mol_m2_s"] == reference_flux
    assert summary["min_n2_crossover_flux_mol_m2_s"] < reference_flux
    assert summary["mean_n2_crossover_flux_mol_m2_s"] < reference_flux
    assert abs(summary["n2_balance_error_mol"]) < 1.0e-12



def test_state_dependent_permeance_model_is_used():
    closure = (
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0020, 0.0020]),
    )

    def permeance_model(anode_rh: float) -> float:
        return 1.0e-11 * (1.0 + anode_rh)

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
        n2_crossover_flux_mol_m2_s=0.0,
        hydrogen_feedback_exponent=0.0,
        cathode_n2_partial_pressure_pa=80000.0,
        n2_permeance_model=permeance_model,
    )

    assert summary["n2_crossover_mode"] == "state_dependent_permeance"
    assert summary["min_n2_permeance_mol_m2_s_pa"] is not None
    assert summary["max_n2_permeance_mol_m2_s_pa"] is not None
    assert (
        summary["max_n2_permeance_mol_m2_s_pa"]
        >= summary["min_n2_permeance_mol_m2_s_pa"]
    )
    assert abs(summary["n2_balance_error_mol"]) < 1.0e-12
