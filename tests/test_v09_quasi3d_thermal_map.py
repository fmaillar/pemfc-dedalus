"""Tests for V0.9 nominal quasi-3D thermal map."""

from __future__ import annotations

import pytest


def test_relative_span_definition() -> None:
    inlet = 5.0
    outlet = 4.5
    span = (inlet - outlet) / inlet
    assert span == pytest.approx(0.1)
