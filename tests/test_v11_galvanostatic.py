"""Tests for V11 galvanostatic electrochemistry."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_galvanostatic import (
    current_density_a_m2,
    faraday_rates_per_cell,
    inverse_symmetric_butler_volmer_overpotential_v,
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
