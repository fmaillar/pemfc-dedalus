from scripts.run_v06_rha_sensitivity import (
    DEFAULT_RH_ANODE_VALUES,
    REGIMES,
    can_reuse_zero_rh_reference,
    case_name,
    v06_result_to_summary,
)


def test_default_anode_rh_values_cover_dry_to_wet():
    assert DEFAULT_RH_ANODE_VALUES == (0.0, 0.10, 0.30, 0.50, 0.70, 0.90)
    assert set(REGIMES) == {"dry_high_load", "nominal", "wet_low_load"}


def test_rha_case_name_is_stable():
    assert case_name("nominal", 0.0) == "nominal_rha0p00"
    assert case_name("nominal", 0.5) == "nominal_rha0p50"


def test_zero_rh_reference_is_reused_only_for_exact_reference_configuration():
    assert can_reuse_zero_rh_reference(
        rh_anode=0.0,
        k_value=2.0e-6,
        nx=16,
        ny=16,
        nz=48,
        membrane_nz=129,
        stop_time=0.0064,
        max_dt=1.0e-6,
        scalar_dt=1.0e-5,
    )
    assert not can_reuse_zero_rh_reference(
        rh_anode=0.1,
        k_value=2.0e-6,
        nx=16,
        ny=16,
        nz=48,
        membrane_nz=129,
        stop_time=0.0064,
        max_dt=1.0e-6,
        scalar_dt=1.0e-5,
    )
    assert not can_reuse_zero_rh_reference(
        rh_anode=0.0,
        k_value=5.0e-6,
        nx=16,
        ny=16,
        nz=48,
        membrane_nz=129,
        stop_time=0.0064,
        max_dt=1.0e-6,
        scalar_dt=1.0e-5,
    )


def test_rha_summary_records_equilibrium_hydration_and_last_iteration():
    data = {
        "converged": True,
        "coupling_iterations": 2,
        "relative_humidity": 0.5,
        "cathode_solid_potential_v": 0.768,
        "lambda_anode_equilibrium": 3.0,
        "anode_water_transfer_coefficient_m_s": 2.0e-6,
        "history": [
            {
                "total_reaction_current_a": 0.003,
                "current_density_a_m2": 1500.0,
                "phi_m_bc_v": -0.02,
                "target_phi_m_bc_v": -0.03,
                "membrane_asr_ohm_m2": 2.0e-5,
                "lambda_anode": 3.2,
                "lambda_mean": 3.3,
                "lambda_cathode": 3.5,
                "sigma_m_mean_s_m": 1.5,
                "sigma_m_min_s_m": 1.4,
                "sigma_m_max_s_m": 1.7,
                "anode_water_removal_flux_lambda_m_s": 4.0e-7,
                "current_relative_change": None,
                "potential_change_v": 0.01,
            },
            {
                "total_reaction_current_a": 0.002,
                "current_density_a_m2": 1000.0,
                "phi_m_bc_v": -0.04,
                "target_phi_m_bc_v": -0.041,
                "membrane_asr_ohm_m2": 4.0e-5,
                "lambda_anode": 2.8,
                "lambda_mean": 3.1,
                "lambda_cathode": 3.5,
                "sigma_m_mean_s_m": 1.2,
                "sigma_m_min_s_m": 0.9,
                "sigma_m_max_s_m": 1.7,
                "anode_water_removal_flux_lambda_m_s": -4.0e-7,
                "current_relative_change": 0.001,
                "potential_change_v": 1.0e-5,
            },
        ],
    }

    summary = v06_result_to_summary("nominal", 0.5, data, "test")

    assert summary["source"] == "test"
    assert summary["lambda_anode_equilibrium"] == 3.0
    assert summary["final_current_a"] == 0.002
    assert summary["final_lambda_anode"] == 2.8
    assert summary["final_anode_water_removal_flux_lambda_m_s"] == -4.0e-7
