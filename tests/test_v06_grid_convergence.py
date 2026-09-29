from scripts.run_v06_grid_convergence import (
    GRIDS,
    QUICK_GRIDS,
    REGIMES,
    can_reuse_reference,
    case_name,
    rel_change,
    v06_result_to_summary,
)


def test_grid_definitions_and_default_regimes_are_stable():
    assert GRIDS == ((16, 16, 48), (24, 24, 72), (32, 32, 96))
    assert QUICK_GRIDS == ((8, 8, 24), (12, 12, 36))
    assert set(REGIMES) == {"dry_high_load", "nominal", "wet_low_load"}


def test_grid_case_name_is_stable():
    assert case_name("dry_high_load", (24, 24, 72)) == "dry_high_load_24x24x72"


def test_relative_change_uses_new_value_as_scale():
    assert rel_change(2.0, 1.0) == 0.5
    assert rel_change(1.0, 1.0) == 0.0


def test_full_reference_is_reused_only_for_exact_reference_configuration():
    assert can_reuse_reference(
        quick=False,
        grid=(16, 16, 48),
        membrane_nz=129,
        stop_time=0.0064,
        max_dt=1.0e-6,
        scalar_dt=1.0e-5,
        anode_relative_humidity=0.0,
        k_value=2.0e-6,
    )
    assert not can_reuse_reference(
        quick=False,
        grid=(24, 24, 72),
        membrane_nz=129,
        stop_time=0.0064,
        max_dt=1.0e-6,
        scalar_dt=1.0e-5,
        anode_relative_humidity=0.0,
        k_value=2.0e-6,
    )
    assert not can_reuse_reference(
        quick=False,
        grid=(16, 16, 48),
        membrane_nz=129,
        stop_time=0.0064,
        max_dt=1.0e-6,
        scalar_dt=1.0e-5,
        anode_relative_humidity=0.0,
        k_value=5.0e-6,
    )
    assert not can_reuse_reference(
        quick=True,
        grid=(16, 16, 48),
        membrane_nz=129,
        stop_time=0.0064,
        max_dt=1.0e-6,
        scalar_dt=1.0e-5,
        anode_relative_humidity=0.0,
        k_value=2.0e-6,
    )


def test_v06_grid_summary_uses_last_coupling_iteration():
    data = {
        "converged": True,
        "coupling_iterations": 2,
        "history": [
            {
                "total_reaction_current_a": 0.003,
                "current_density_a_m2": 1500.0,
                "phi_m_bc_v": -0.02,
                "target_phi_m_bc_v": -0.03,
                "membrane_asr_ohm_m2": 2.0e-5,
                "lambda_anode": 3.0,
                "lambda_mean": 3.2,
                "lambda_cathode": 3.5,
                "sigma_m_mean_s_m": 1.5,
                "sigma_m_min_s_m": 1.4,
                "sigma_m_max_s_m": 1.7,
                "anode_water_removal_flux_lambda_m_s": 2.0e-6,
                "current_relative_change": None,
                "potential_change_v": 0.01,
            },
            {
                "total_reaction_current_a": 0.002,
                "current_density_a_m2": 1000.0,
                "phi_m_bc_v": -0.04,
                "target_phi_m_bc_v": -0.041,
                "membrane_asr_ohm_m2": 4.0e-5,
                "lambda_anode": 2.0,
                "lambda_mean": 2.5,
                "lambda_cathode": 3.5,
                "sigma_m_mean_s_m": 1.2,
                "sigma_m_min_s_m": 0.9,
                "sigma_m_max_s_m": 1.7,
                "anode_water_removal_flux_lambda_m_s": 4.0e-6,
                "current_relative_change": 0.001,
                "potential_change_v": 1.0e-5,
            },
        ],
    }

    summary = v06_result_to_summary(data, "test")

    assert summary["source"] == "test"
    assert summary["converged"] is True
    assert summary["final_current_a"] == 0.002
    assert summary["final_lambda_anode"] == 2.0
    assert summary["final_anode_water_removal_flux_lambda_m_s"] == 4.0e-6
    assert summary["final_current_relative_change"] == 0.001
