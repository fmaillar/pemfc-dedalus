"""Tests for finite-NTU Dedalus streamwise heating."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.cathode_airflow_1d import (
    solve_streamwise_finite_thermal_profile,
)
from pemfc_dedalus.parameters import CathodeParameters


def test_finite_thermal_profile_matches_exponential_solution() -> None:
    p = CathodeParameters()
    current_a = 26.04
    ntu = 3.0
    inlet_temperature_k = 293.15
    stack_temperature_k = 313.15

    solved = solve_streamwise_finite_thermal_profile(
        current_a=current_a,
        total_air_flow_slpm=p.stack.coolant_air_target_slpm(current_a),
        n_cells=p.stack.n_cells,
        oxygen_mole_fraction=p.oxygen_mole_fraction,
        faraday_c_mol=p.faraday,
        inlet_temperature_k=inlet_temperature_k,
        stack_temperature_k=stack_temperature_k,
        ntu=ntu,
        points=48,
    )
    expected = stack_temperature_k - (
        stack_temperature_k - inlet_temperature_k
    ) * np.exp(-ntu * solved.streamwise_fraction)

    assert np.max(np.abs(solved.air_temperature_k - expected)) < 1.0e-9
    assert solved.air_temperature_k[-1] < stack_temperature_k


def test_zero_ntu_keeps_inlet_temperature() -> None:
    p = CathodeParameters()
    solved = solve_streamwise_finite_thermal_profile(
        current_a=7.3,
        total_air_flow_slpm=p.stack.coolant_air_target_slpm(7.3),
        n_cells=p.stack.n_cells,
        oxygen_mole_fraction=p.oxygen_mole_fraction,
        faraday_c_mol=p.faraday,
        inlet_temperature_k=293.15,
        stack_temperature_k=303.15,
        ntu=0.0,
        points=32,
    )
    assert np.allclose(solved.air_temperature_k, 293.15)


def test_negative_ntu_is_rejected() -> None:
    p = CathodeParameters()
    with pytest.raises(ValueError, match="ntu must be non-negative"):
        solve_streamwise_finite_thermal_profile(
            current_a=7.3,
            total_air_flow_slpm=p.stack.coolant_air_target_slpm(7.3),
            n_cells=p.stack.n_cells,
            oxygen_mole_fraction=p.oxygen_mole_fraction,
            faraday_c_mol=p.faraday,
            inlet_temperature_k=293.15,
            stack_temperature_k=303.15,
            ntu=-1.0,
        )
