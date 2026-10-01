"""Tests for V0.9 final validation helpers."""

from __future__ import annotations

from scripts.analyze_v09_final_validation import _assert_monotonic


def test_monotonic_helper_accepts_expected_order() -> None:
    cases = [
        {
            "current_a": 1.0,
            "inlet_temperature_c": 20.0,
            "ntu": 3.0,
            "reaction_current_span": 0.01,
        },
        {
            "current_a": 2.0,
            "inlet_temperature_c": 20.0,
            "ntu": 3.0,
            "reaction_current_span": 0.02,
        },
    ]
    _assert_monotonic(
        cases,
        group_keys=("inlet_temperature_c", "ntu"),
        sort_key="current_a",
        increasing=True,
    )
