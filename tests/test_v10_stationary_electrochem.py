"""Tests for V1.0 stationary electrochemistry helpers."""

from __future__ import annotations

import pytest

from scripts.run_v10_stationary_electrochem import simpson_mean


def test_simpson_mean() -> None:
    assert simpson_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_simpson_mean_requires_three_values() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        simpson_mean([1.0, 2.0])
