"""Tests for V1.0 common-voltage bisection utilities."""

from __future__ import annotations

import pytest

from scripts.run_v10_voltage_bisection import simpson_mean


def test_simpson_mean() -> None:
    assert simpson_mean([2.0, 4.0, 6.0]) == pytest.approx(4.0)


def test_simpson_mean_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        simpson_mean([2.0, 4.0])
