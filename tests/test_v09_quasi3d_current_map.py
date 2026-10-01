"""Tests for current-specific cathode electrochemistry parameters."""

from __future__ import annotations

from dataclasses import replace

from pemfc_dedalus.parameters import CathodeParameters


def test_current_specific_parameters_change_thermal_and_voltage_state() -> None:
    base = CathodeParameters()
    current_a = 14.5
    target_temperature_k = 273.15 + base.tech.optimum_stack_temperature_c(current_a)
    cell_voltage_v = base.tech.bol_typical_cell_voltage_v(current_a)

    updated = replace(
        base,
        stack_current_a=current_a,
        stack_temperature=target_temperature_k,
        cathode_solid_potential=cell_voltage_v,
    )

    assert updated.stack_current_a == current_a
    assert updated.stack_temperature == target_temperature_k
    assert updated.cathode_solid_potential == cell_voltage_v
    assert updated.beta_cathodic != base.beta_cathodic
