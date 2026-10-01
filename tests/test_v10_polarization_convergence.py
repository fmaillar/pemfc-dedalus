"""Tests for V1.0 polarization convergence utilities."""

from __future__ import annotations

import pytest

from scripts.run_v10_polarization_convergence import simpson_mean


def test_simpson_mean_quadratic_samples() -> None:
    assert simpson_mean([0.0, 0.25, 1.0]) == pytest.approx(1.0 / 3.0)


def test_simpson_mean_requires_three_values() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        simpson_mean([1.0])
