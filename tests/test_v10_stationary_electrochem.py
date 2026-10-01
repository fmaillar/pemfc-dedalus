"""Smoke-test the V1.0 stationary electrochemistry screening module."""

from __future__ import annotations

from scripts import run_v10_stationary_electrochem


def test_stationary_screening_module_imports() -> None:
    assert callable(run_v10_stationary_electrochem.main)
