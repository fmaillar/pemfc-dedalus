"""Tests for V0.9 quasi-3D convergence utilities."""

from __future__ import annotations

import pytest


def test_reaction_current_span_definition() -> None:
    inlet = 10.0
    outlet = 9.3
    span = (inlet - outlet) / inlet
    assert span == pytest.approx(0.07)
