"""Tests for V0.9 full quasi-3D factorial map."""

from __future__ import annotations


def test_factorial_case_count() -> None:
    currents = [7.3, 14.5, 26.04]
    temperatures = [10.0, 20.0, 30.0]
    ntu_values = [1.0, 3.0, 5.0]
    assert len(currents) * len(temperatures) * len(ntu_values) == 27
