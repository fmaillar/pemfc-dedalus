import pytest

from pemfc_dedalus.gas_permeability import (
    BARRER_TO_MOL_M_PER_M2_S_PA,
    arrhenius_permeability,
    barrer_to_si_permeability,
    effective_membrane_relative_humidity,
    humidity_permeability_multiplier,
    membrane_permeance_from_permeability,
    state_dependent_permeability,
)


def test_barrer_conversion_matches_definition():
    assert barrer_to_si_permeability(1.0) == pytest.approx(
        BARRER_TO_MOL_M_PER_M2_S_PA
    )


def test_membrane_permeance_scales_inversely_with_thickness():
    permeability = barrer_to_si_permeability(0.24)
    permeance_50um = membrane_permeance_from_permeability(
        permeability,
        50.0e-6,
    )
    permeance_100um = membrane_permeance_from_permeability(
        permeability,
        100.0e-6,
    )
    assert permeance_50um == pytest.approx(2.0 * permeance_100um)


def test_gas_permeability_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        barrer_to_si_permeability(-1.0)
    with pytest.raises(ValueError):
        membrane_permeance_from_permeability(-1.0, 50.0e-6)
    with pytest.raises(ValueError):
        membrane_permeance_from_permeability(1.0, 0.0)



def test_arrhenius_permeability_increases_with_temperature():
    reference = barrer_to_si_permeability(0.24)
    at_reference = arrhenius_permeability(
        reference,
        308.15,
        308.15,
        19_830.0,
        8.31446261815324,
    )
    warmer = arrhenius_permeability(
        reference,
        318.15,
        308.15,
        19_830.0,
        8.31446261815324,
    )
    assert at_reference == pytest.approx(reference)
    assert warmer > reference


def test_humidity_multiplier_is_bounded_and_nonlinear():
    assert humidity_permeability_multiplier(0.0) == 1.0
    assert humidity_permeability_multiplier(0.9) == pytest.approx(100.0)
    assert humidity_permeability_multiplier(1.0) == pytest.approx(100.0)
    assert 1.0 < humidity_permeability_multiplier(0.5) < 100.0


def test_effective_membrane_rh_is_mean_boundary_rh():
    assert effective_membrane_relative_humidity(0.2, 0.8) == pytest.approx(0.5)



def test_state_dependent_permeability_combines_temperature_and_humidity():
    reference = barrer_to_si_permeability(0.24)
    value = state_dependent_permeability(
        reference,
        temperature_k=313.15,
        reference_temperature_k=308.15,
        activation_energy_j_mol=19_830.0,
        gas_constant_j_mol_k=8.31446261815324,
        anode_relative_humidity=0.5,
        cathode_relative_humidity=0.5,
        maximum_humidity_factor=100.0,
        humidity_reference_relative_humidity=0.9,
        humidity_shape_exponent=2.0,
    )
    assert value > reference
