import csv
from pathlib import Path

import pytest

from scripts.run_v06_ka_sensitivity import (
    DEFAULT_K_VALUES,
    REGIMES,
    case_name,
    load_v05_reference_rows,
    v05_reference_to_summary,
    v06_result_to_summary,
)


def test_default_ka_sensitivity_values_span_zero_and_reference():
    assert DEFAULT_K_VALUES == (0.0, 5.0e-7, 1.0e-6, 2.0e-6, 5.0e-6, 1.0e-5)
    assert 2.0e-6 in DEFAULT_K_VALUES
    assert set(REGIMES) == {"dry_high_load", "nominal", "wet_low_load"}


def test_ka_case_name_is_stable_and_distinct():
    assert case_name("nominal", 5.0e-7) == "nominal_ka5p0em07"
    assert case_name("nominal", 1.0e-6) == "nominal_ka1p0em06"
    assert case_name("nominal", 5.0e-7) != case_name("nominal", 1.0e-6)


def test_v05_reference_is_loaded_as_zero_transfer(tmp_path: Path):
    path = tmp_path / "reference.csv"
    fields = [
        "rh",
        "cathode_solid_potential_v",
        "converged",
        "coupling_iterations",
        "final_current_a",
        "final_current_density_a_m2",
        "final_phi_m_bc_v",
        "final_target_phi_m_bc_v",
        "final_membrane_asr_ohm_m2",
        "final_lambda_anode",
        "final_lambda_mean",
        "final_lambda_cathode",
        "final_sigma_m_mean_s_m",
        "final_sigma_m_min_s_m",
        "final_sigma_m_max_s_m",
    ]
    values = [
        "0.5",
        "0.768",
        "True",
        "5",
        "0.002",
        "1000.0",
        "-0.03",
        "-0.031",
        "3e-5",
        "3.2",
        "3.3",
        "3.5",
        "1.6",
        "1.5",
        "1.7",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerow(values)

    rows = load_v05_reference_rows(path)
    summary = v05_reference_to_summary("nominal", 0.0, rows[(0.5, 0.768)])

    assert summary["source"] == "v05_zero_transfer_reference"
    assert summary["converged"] is True
    assert summary["final_current_a"] == pytest.approx(0.002)
    assert summary["final_anode_water_removal_flux_lambda_m_s"] == 0.0


def test_v06_result_summary_uses_last_coupling_iteration():
    data = {
        "converged": True,
        "coupling_iterations": 2,
        "relative_humidity": 0.5,
        "cathode_solid_potential_v": 0.768,
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

    summary = v06_result_to_summary("nominal", 2.0e-6, data, "test")

    assert summary["final_current_a"] == pytest.approx(0.002)
    assert summary["final_lambda_anode"] == pytest.approx(2.0)
    assert summary["final_anode_water_removal_flux_lambda_m_s"] == pytest.approx(4.0e-6)
    assert summary["final_current_relative_change"] == pytest.approx(0.001)
