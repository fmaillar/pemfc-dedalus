"""Tests for the V0.9 Dedalus streamwise verification model."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.cathode_airflow import (
    streamwise_air_temperature_profile,
    streamwise_oxygen_profile,
)
from pemfc_dedalus.cathode_airflow_1d import solve_streamwise_coupled_profile
from pemfc_dedalus.parameters import CathodeParameters


def test_dedalus_streamwise_solution_matches_analytic_balances() -> None:
    p = CathodeParameters()
    current_a = 26.04
    total_flow_slpm = p.stack.coolant_air_target_slpm(current_a)
    vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
    heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
    inlet_temperature_k = 283.15

    solved = solve_streamwise_coupled_profile(
        current_a=current_a,
        total_air_flow_slpm=total_flow_slpm,
        n_cells=p.stack.n_cells,
        oxygen_mole_fraction=p.oxygen_mole_fraction,
        faraday_c_mol=p.faraday,
        inlet_temperature_k=inlet_temperature_k,
        total_heat_rejection_w=heat_rejection_w,
        points=32,
    )
    analytic_o2 = streamwise_oxygen_profile(
        current_a=current_a,
        total_air_flow_slpm=total_flow_slpm,
        n_cells=p.stack.n_cells,
        oxygen_mole_fraction=p.oxygen_mole_fraction,
        faraday_c_mol=p.faraday,
        points=129,
    )
    analytic_t = streamwise_air_temperature_profile(
        inlet_temperature_k=inlet_temperature_k,
        total_air_flow_slpm=total_flow_slpm,
        n_cells=p.stack.n_cells,
        total_heat_rejection_w=heat_rejection_w,
        points=129,
    )

    expected_o2 = np.interp(
        solved.streamwise_fraction,
        analytic_o2.streamwise_fraction,
        analytic_o2.oxygen_molar_flow_mol_s,
    )
    expected_t = np.interp(
        solved.streamwise_fraction,
        analytic_t.streamwise_fraction,
        analytic_t.air_temperature_k,
    )

    assert np.max(
        np.abs(solved.oxygen_molar_flow_mol_s - expected_o2)
    ) < 1.0e-11
    assert np.max(np.abs(solved.air_temperature_k - expected_t)) < 1.0e-9
    assert solved.air_temperature_k[-1] > solved.air_temperature_k[0]
    assert solved.oxygen_mole_fraction[-1] < solved.oxygen_mole_fraction[0]


def test_dedalus_streamwise_rejects_too_small_grid() -> None:
    p = CathodeParameters()
    with pytest.raises(ValueError, match="points must be >= 4"):
        solve_streamwise_coupled_profile(
            current_a=1.0,
            total_air_flow_slpm=10.0,
            n_cells=p.stack.n_cells,
            oxygen_mole_fraction=p.oxygen_mole_fraction,
            faraday_c_mol=p.faraday,
            inlet_temperature_k=293.15,
            total_heat_rejection_w=1.0,
            points=3,
        )
