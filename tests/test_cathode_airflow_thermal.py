"""Tests for V0.9 streamwise cathode thermal profiles."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.cathode_airflow import streamwise_air_temperature_profile
from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.thermal import outlet_temperature_k


def test_streamwise_thermal_profile_matches_lumped_outlet_balance() -> None:
    p = CathodeParameters()
    current_a = 26.04
    vcell_v = p.tech.bol_typical_cell_voltage_v(current_a)
    heat_rejection_w = p.stack.heat_rejection_w(current_a, vcell_v)
    total_flow = p.stack.coolant_air_target_slpm(current_a)
    inlet_temperature_k = 283.15

    profile = streamwise_air_temperature_profile(
        inlet_temperature_k=inlet_temperature_k,
        total_air_flow_slpm=total_flow,
        n_cells=p.stack.n_cells,
        total_heat_rejection_w=heat_rejection_w,
        points=51,
    )
    expected = outlet_temperature_k(
        heat_rejection_w=heat_rejection_w,
        inlet_temperature_k=inlet_temperature_k,
        air_flow_slpm=total_flow,
    )
    assert profile.outlet_temperature_k == pytest.approx(expected)


def test_streamwise_thermal_profile_is_monotonic() -> None:
    profile = streamwise_air_temperature_profile(
        inlet_temperature_k=293.15,
        total_air_flow_slpm=100.0,
        n_cells=10,
        total_heat_rejection_w=50.0,
        points=17,
    )
    assert np.all(np.diff(profile.air_temperature_k) >= 0.0)
    assert profile.temperature_rise_k > 0.0


def test_zero_heat_keeps_temperature_uniform() -> None:
    profile = streamwise_air_temperature_profile(
        inlet_temperature_k=293.15,
        total_air_flow_slpm=100.0,
        n_cells=10,
        total_heat_rejection_w=0.0,
        points=17,
    )
    assert np.allclose(profile.air_temperature_k, 293.15)
    assert profile.temperature_rise_k == pytest.approx(0.0)
