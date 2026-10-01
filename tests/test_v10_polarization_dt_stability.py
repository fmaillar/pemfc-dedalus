"""Tests for V1.0 timestep-stability utilities."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_v10_polarization_dt_stability import simpson_mean


def test_simpson_mean_vectorized() -> None:
    values = np.array([[2.0, 4.0], [3.0, 5.0], [4.0, 6.0]])
    result = simpson_mean(values)
    assert result.tolist() == pytest.approx([3.0, 5.0])


def test_simpson_mean_requires_three_rows() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        simpson_mean(np.ones((2, 4)))
