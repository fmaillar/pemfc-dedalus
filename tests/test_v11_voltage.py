"""Tests for the predictive V11 cell-voltage closure."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_voltage import predict_cell_voltage_v


def test_open_circuit_voltage_has_no_activation_or_ohmic_loss() -> None:
    result = predict_cell_voltage_v(
        current_a=0.0,
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=136000.0,
        oxygen_partial_pressure_pa=21000.0,
        water_activity=1.0,
        membrane_mean_water_content=5.0,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )

    assert result.activation_loss_v == pytest.approx(0.0)
    assert result.membrane_ohmic_loss_v == pytest.approx(0.0)
    assert result.cell_voltage_v == pytest.approx(result.reversible_v)


def test_loaded_voltage_is_below_reversible_voltage() -> None:
    result = predict_cell_voltage_v(
        current_a=26.04,
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=136000.0,
        oxygen_partial_pressure_pa=21000.0,
        water_activity=1.0,
        membrane_mean_water_content=5.0,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )

    assert result.activation_loss_v > 0.0
    assert result.membrane_ohmic_loss_v > 0.0
    assert result.cell_voltage_v < result.reversible_v


def test_more_platinum_reduces_activation_loss() -> None:
    common = {
        "current_a": 26.04,
        "temperature_k": 313.15,
        "hydrogen_partial_pressure_pa": 136000.0,
        "oxygen_partial_pressure_pa": 21000.0,
        "water_activity": 1.0,
        "membrane_mean_water_content": 5.0,
        "cathode_ecsa_m2_pt_g_pt": 50.0,
    }
    low_pt = predict_cell_voltage_v(
        cathode_platinum_loading_mg_cm2_geo=0.2,
        **common,
    )
    high_pt = predict_cell_voltage_v(
        cathode_platinum_loading_mg_cm2_geo=0.6,
        **common,
    )

    assert high_pt.activation_loss_v < low_pt.activation_loss_v
    assert high_pt.cell_voltage_v > low_pt.cell_voltage_v


def test_better_ecsa_reduces_activation_loss() -> None:
    low = predict_cell_voltage_v(
        current_a=26.04,
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=136000.0,
        oxygen_partial_pressure_pa=21000.0,
        water_activity=1.0,
        membrane_mean_water_content=5.0,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=30.0,
    )
    high = predict_cell_voltage_v(
        current_a=26.04,
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=136000.0,
        oxygen_partial_pressure_pa=21000.0,
        water_activity=1.0,
        membrane_mean_water_content=5.0,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=70.0,
    )

    assert high.activation_loss_v < low.activation_loss_v


def test_additional_resolved_loss_is_subtracted_exactly() -> None:
    baseline = predict_cell_voltage_v(
        current_a=26.04,
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=136000.0,
        oxygen_partial_pressure_pa=21000.0,
        water_activity=1.0,
        membrane_mean_water_content=5.0,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )
    extra = predict_cell_voltage_v(
        current_a=26.04,
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=136000.0,
        oxygen_partial_pressure_pa=21000.0,
        water_activity=1.0,
        membrane_mean_water_content=5.0,
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
        additional_resolved_loss_v=0.05,
    )

    assert extra.cell_voltage_v == pytest.approx(
        baseline.cell_voltage_v - 0.05
    )
