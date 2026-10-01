"""Tests for V11 galvanostatic electrochemistry."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_galvanostatic import (
    current_density_a_m2,
    faraday_rates_per_cell,
    inverse_symmetric_butler_volmer_overpotential_v,
    membrane_ohmic_loss_v,
    resolved_cell_voltage_v,
    reversible_cell_voltage_liquid_water_v,
    stack_terminal_voltage_v,
    symmetric_butler_volmer_current_density_a_m2,
)


def test_faraday_rates_preserve_pemfc_stoichiometry() -> None:
    rates = faraday_rates_per_cell(26.04)

    assert rates.hydrogen_consumption_mol_s > 0.0
    assert rates.oxygen_consumption_mol_s > 0.0
    assert rates.water_production_mol_s > 0.0
    assert rates.hydrogen_consumption_mol_s == pytest.approx(
        2.0 * rates.oxygen_consumption_mol_s
    )
    assert rates.water_production_mol_s == pytest.approx(
        rates.hydrogen_consumption_mol_s
    )


def test_zero_current_has_zero_reaction_rates() -> None:
    rates = faraday_rates_per_cell(0.0)

    assert rates.hydrogen_consumption_mol_s == 0.0
    assert rates.oxygen_consumption_mol_s == 0.0
    assert rates.water_production_mol_s == 0.0


def test_current_density_uses_active_area() -> None:
    assert current_density_a_m2(10.0, 0.01) == pytest.approx(1000.0)


def test_inverse_symmetric_bv_round_trip() -> None:
    current_density = 2150.0
    exchange_current_density = 3.2
    alpha = 0.5
    temperature = 313.15
    activity = 0.72

    eta = inverse_symmetric_butler_volmer_overpotential_v(
        current_density_a_m2=current_density,
        exchange_current_density_a_m2=exchange_current_density,
        transfer_coefficient=alpha,
        temperature_k=temperature,
        reactant_activity=activity,
    )
    recovered = symmetric_butler_volmer_current_density_a_m2(
        activation_loss_v=eta,
        exchange_current_density_a_m2=exchange_current_density,
        transfer_coefficient=alpha,
        temperature_k=temperature,
        reactant_activity=activity,
    )

    assert eta > 0.0
    assert recovered == pytest.approx(current_density, rel=1e-12)


def test_inverse_bv_zero_current_has_zero_activation_loss() -> None:
    eta = inverse_symmetric_butler_volmer_overpotential_v(
        current_density_a_m2=0.0,
        exchange_current_density_a_m2=1.0,
        transfer_coefficient=0.5,
        temperature_k=300.0,
    )

    assert eta == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("current_a", "active_area_m2"),
    [
        (-1.0, 0.01),
        (1.0, 0.0),
        (1.0, -0.01),
    ],
)
def test_current_density_rejects_nonphysical_inputs(
    current_a: float,
    active_area_m2: float,
) -> None:
    with pytest.raises(ValueError):
        current_density_a_m2(current_a, active_area_m2)



def test_reversible_voltage_is_about_1_229_v_at_standard_conditions() -> None:
    voltage = reversible_cell_voltage_liquid_water_v(
        temperature_k=298.15,
        hydrogen_partial_pressure_pa=1.0e5,
        oxygen_partial_pressure_pa=1.0e5,
    )

    assert voltage == pytest.approx(1.229, abs=2.0e-3)


def test_reversible_voltage_increases_with_reactant_pressure() -> None:
    base = reversible_cell_voltage_liquid_water_v(
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=1.0e5,
        oxygen_partial_pressure_pa=0.21e5,
    )
    pressurized = reversible_cell_voltage_liquid_water_v(
        temperature_k=313.15,
        hydrogen_partial_pressure_pa=1.36e5,
        oxygen_partial_pressure_pa=0.21e5,
    )

    assert pressurized > base


def test_membrane_ohmic_loss_is_j_l_over_sigma() -> None:
    loss = membrane_ohmic_loss_v(
        current_density_a_m2=2000.0,
        membrane_thickness_m=50e-6,
        proton_conductivity_s_m=10.0,
    )

    assert loss == pytest.approx(0.01)



def test_resolved_cell_voltage_is_explicit_sum_of_losses() -> None:
    breakdown = resolved_cell_voltage_v(
        reversible_v=1.20,
        activation_loss_v=0.30,
        membrane_ohmic_loss_v=0.05,
        other_resolved_loss_v=0.02,
    )

    assert breakdown.cell_voltage_v == pytest.approx(0.83)


def test_stack_terminal_voltage_includes_bus_plate_loss() -> None:
    voltage = stack_terminal_voltage_v(
        cell_voltage_v=0.768,
        n_cells=10,
        current_a=26.04,
        bus_plate_resistance_ohm=2.2e-3,
    )

    assert voltage == pytest.approx(7.622712)


def test_resolved_voltage_rejects_negative_loss() -> None:
    with pytest.raises(ValueError):
        resolved_cell_voltage_v(
            reversible_v=1.20,
            activation_loss_v=-0.01,
            membrane_ohmic_loss_v=0.05,
        )
