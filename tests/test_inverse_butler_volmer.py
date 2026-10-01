"""Tests for the exact inverse symmetric Butler-Volmer relation."""

from __future__ import annotations

import math

import pytest

from pemfc_dedalus.cathode_electrochem_inverse_bv_3d import (
    inverse_symmetric_bv_eta,
    symmetric_bv_current,
)
from pemfc_dedalus.parameters import CathodeParameters


def test_inverse_symmetric_bv_round_trip_nominal_cathodic_branch() -> None:
    params = CathodeParameters()
    eta_v = -0.41183482
    oxygen_activity = 7.5715 / params.oxygen_inlet_concentration
    exchange_current = params.j0_vol * oxygen_activity**params.oxygen_reaction_order

    current = symmetric_bv_current(
        eta_v,
        exchange_current,
        params.beta_cathodic,
    )
    recovered_eta = inverse_symmetric_bv_eta(
        current,
        exchange_current,
        params.beta_cathodic,
    )

    assert recovered_eta == pytest.approx(eta_v, rel=1e-12, abs=1e-14)


def test_inverse_symmetric_bv_is_regular_at_zero_current() -> None:
    params = CathodeParameters()

    eta = inverse_symmetric_bv_eta(
        0.0,
        params.j0_vol,
        params.beta_cathodic,
    )

    assert eta == pytest.approx(0.0, abs=1e-15)


def test_anodic_term_is_negligible_at_nominal_cathodic_overpotential() -> None:
    params = CathodeParameters()
    eta_v = -0.41183482

    anodic_to_cathodic = math.exp(
        (params.beta_anodic + params.beta_cathodic) * eta_v
    )

    assert anodic_to_cathodic < 1e-6
