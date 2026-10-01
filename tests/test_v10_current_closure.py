"""Tests for V1.0 current-closure screening."""

from __future__ import annotations

import pytest

from scripts.analyze_v10_current_closure import target_current_density_a_m2


def test_target_current_density_scales_linearly() -> None:
    value = target_current_density_a_m2(
        current_a=13.02,
        reference_current_a=26.04,
        reference_current_density_a_m2=2150.0,
    )
    assert value == pytest.approx(1075.0)


def test_target_current_density_rejects_nonpositive_current() -> None:
    with pytest.raises(ValueError, match="current_a"):
        target_current_density_a_m2(
            current_a=0.0,
            reference_current_a=26.04,
            reference_current_density_a_m2=2150.0,
        )
