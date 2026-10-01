"""Tests for V0.9 electrochemical slice projection."""

from __future__ import annotations

import numpy as np

from pemfc_dedalus.parameters import CathodeParameters


def test_linear_oxygen_reaction_order_preserves_activity_factor() -> None:
    p = CathodeParameters()
    activities = np.array([0.9, 1.0, 1.1])
    factors = activities ** p.oxygen_reaction_order
    assert np.allclose(factors, activities)
