"""Tests for V1.0 stationary-window diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_v10_polarization_stationary import (
    _find_stable_window,
    simpson_mean,
)


def test_simpson_mean_vectorized() -> None:
    values = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    result = simpson_mean(values)
    assert result.tolist() == pytest.approx([2.0, 3.0])


def test_find_stable_window_detects_plateau() -> None:
    times = np.arange(10, dtype=float)
    values = np.array([10, 9, 8, 7, 6, 5.01, 5.00, 5.01, 5.00, 5.01])
    result = _find_stable_window(
        times,
        values,
        window_points=4,
        relative_tolerance=5e-3,
        skip_fraction=0.2,
    )
    assert result is not None
    assert result["start_time_s"] >= 5.0
