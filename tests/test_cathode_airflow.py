"""Tests for reduced V0.9 open-cathode airflow balances."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.ballard_1020acs import Ballard1020ACSTechnologyReference
from pemfc_dedalus.cathode_airflow import (
    molar_flow_from_slpm,
    oxygen_consumption_mol_s,
    streamwise_oxygen_profile,
)
from pemfc_dedalus.parameters import CathodeParameters


def test_manual_air_correlation_matches_faraday_oxygen_demand() -> None:
    p = CathodeParameters()
    current_a = 26.04
    flow_slpm = p.tech.stoichiometric_air_slpm_per_cell(current_a)
    oxygen_supply = (
        molar_flow_from_slpm(flow_slpm) * p.oxygen_mole_fraction
    )
    oxygen_demand = oxygen_consumption_mol_s(
        current_a,
        faraday_c_mol=p.faraday,
    )
    assert oxygen_supply == pytest.approx(oxygen_demand, rel=3.0e-3)


def test_streamwise_profile_conserves_oxygen() -> None:
    p = CathodeParameters()
    current_a = 26.04
    total_flow = p.stack.coolant_air_target_slpm(current_a)
    profile = streamwise_oxygen_profile(
        current_a=current_a,
        total_air_flow_slpm=total_flow,
        n_cells=p.stack.n_cells,
        oxygen_mole_fraction=p.oxygen_mole_fraction,
        faraday_c_mol=p.faraday,
        points=51,
    )
    consumed = oxygen_consumption_mol_s(
        current_a,
        faraday_c_mol=p.faraday,
    )
    delta = (
        profile.oxygen_molar_flow_mol_s[0]
        - profile.oxygen_molar_flow_mol_s[-1]
    )
    assert delta == pytest.approx(consumed)
    assert np.all(np.diff(profile.oxygen_mole_fraction) <= 0.0)


def test_no_current_keeps_uniform_oxygen_fraction() -> None:
    p = CathodeParameters()
    profile = streamwise_oxygen_profile(
        current_a=0.0,
        total_air_flow_slpm=10.0,
        n_cells=p.stack.n_cells,
        oxygen_mole_fraction=p.oxygen_mole_fraction,
        faraday_c_mol=p.faraday,
        points=17,
    )
    assert np.allclose(
        profile.oxygen_mole_fraction,
        p.oxygen_mole_fraction,
    )
    assert profile.oxygen_utilization == 0.0
    assert profile.oxygen_stoichiometry == float("inf")


def test_insufficient_airflow_is_rejected() -> None:
    p = CathodeParameters()
    current_a = Ballard1020ACSTechnologyReference().current_max_a
    with pytest.raises(ValueError, match="cannot supply"):
        streamwise_oxygen_profile(
            current_a=current_a,
            total_air_flow_slpm=0.1,
            n_cells=p.stack.n_cells,
            oxygen_mole_fraction=p.oxygen_mole_fraction,
            faraday_c_mol=p.faraday,
        )
