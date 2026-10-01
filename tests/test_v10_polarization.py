"""Tests for V1.0 thermo-coupled polarization utilities."""

from __future__ import annotations

import pytest

from scripts.run_v10_polarization import simpson_mean


def test_simpson_mean_constant_profile() -> None:
    assert simpson_mean([5.0, 5.0, 5.0]) == pytest.approx(5.0)


def test_simpson_mean_linear_profile() -> None:
    assert simpson_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
