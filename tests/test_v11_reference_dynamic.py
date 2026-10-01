"""Tests for V11 reference-transient convergence utilities."""

from __future__ import annotations

import pytest

from scripts.run_v11_reference_dynamic import convergence_metrics


def test_convergence_metrics_are_zero_for_identical_rows() -> None:
    rows = [
        {
            "time_s": "0.0",
            "cell_voltage_v": "0.80",
            "stack_temperature_k": "303.15",
            "membrane_mean_water_content": "3.5",
            "anode_nitrogen_mol": "0.0",
        },
        {
            "time_s": "1.0",
            "cell_voltage_v": "0.79",
            "stack_temperature_k": "303.20",
            "membrane_mean_water_content": "3.6",
            "anode_nitrogen_mol": "1.0e-8",
        },
    ]

    metrics = convergence_metrics(
        reference_rows=rows,
        candidate_rows=rows,
    )

    assert all(value == pytest.approx(0.0) for value in metrics.values())


def test_convergence_metrics_interpolate_candidate_grid() -> None:
    reference = [
        {
            "time_s": "0.0",
            "cell_voltage_v": "0.80",
            "stack_temperature_k": "300.0",
            "membrane_mean_water_content": "3.0",
            "anode_nitrogen_mol": "0.0",
        },
        {
            "time_s": "0.5",
            "cell_voltage_v": "0.75",
            "stack_temperature_k": "301.0",
            "membrane_mean_water_content": "4.0",
            "anode_nitrogen_mol": "1.0",
        },
        {
            "time_s": "1.0",
            "cell_voltage_v": "0.70",
            "stack_temperature_k": "302.0",
            "membrane_mean_water_content": "5.0",
            "anode_nitrogen_mol": "2.0",
        },
    ]
    candidate = [reference[0], reference[-1]]

    metrics = convergence_metrics(
        reference_rows=reference,
        candidate_rows=candidate,
    )

    assert metrics["cell_voltage_v_max_abs"] == pytest.approx(0.0)
    assert metrics["stack_temperature_k_max_abs"] == pytest.approx(0.0)
    assert metrics["membrane_mean_water_content_max_abs"] == pytest.approx(0.0)
    assert metrics["anode_nitrogen_mol_max_abs"] == pytest.approx(0.0)
