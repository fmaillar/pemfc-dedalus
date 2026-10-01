"""Tests for V1.0 pseudo-time convergence utilities."""

from __future__ import annotations

import pytest

from scripts.run_v10_polarization_time_convergence import simpson_mean


def test_simpson_mean_linear_profile() -> None:
    assert simpson_mean([2.0, 3.0, 4.0]) == pytest.approx(3.0)


def test_simpson_mean_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        simpson_mean([2.0, 3.0])
