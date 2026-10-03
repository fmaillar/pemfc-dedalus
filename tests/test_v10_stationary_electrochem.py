"""Smoke-test the V1.0 stationary electrochemistry screening module."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.cathode_electrochem_stationary_3d import (
    _linearized_bv_coefficients,
)
from pemfc_dedalus.parameters import CathodeParameters
from scripts import run_v10_stationary_electrochem


def test_stationary_screening_module_imports() -> None:
    assert callable(run_v10_stationary_electrochem.main)


def test_linearized_bv_matches_value_and_slope_at_nominal_eta() -> None:
    params = CathodeParameters()
    eta_ref = (
        params.cathode_solid_potential
        - params.membrane_proton_potential
        - params.equilibrium_potential
    )
    factor, slope = _linearized_bv_coefficients(
        eta_ref=eta_ref,
        beta_a=params.beta_anodic,
        beta_c=params.beta_cathodic,
    )

    expected_factor = np.exp(-params.beta_cathodic * eta_ref) - np.exp(
        params.beta_anodic * eta_ref
    )
    expected_slope = (
        -params.beta_cathodic * np.exp(-params.beta_cathodic * eta_ref)
        - params.beta_anodic * np.exp(params.beta_anodic * eta_ref)
    )

    assert factor == pytest.approx(expected_factor)
    assert slope == pytest.approx(expected_slope)
    assert factor > 0.0
    assert slope < 0.0
