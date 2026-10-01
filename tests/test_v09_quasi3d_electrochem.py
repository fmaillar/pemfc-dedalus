"""Tests for V0.9 local cathode-feed override."""

from __future__ import annotations

from pathlib import Path

import pytest

from pemfc_dedalus.cathode_electrochem_3d import build_solver
from pemfc_dedalus.parameters import CathodeParameters


def test_cathode_solver_rejects_nonpositive_local_feed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="oxygen_feed_concentration"):
        build_solver(
            CathodeParameters(),
            nx=4,
            ny=4,
            nz=8,
            stop_time=1.0e-6,
            output_dir=tmp_path,
            oxygen_feed_concentration=0.0,
        )
