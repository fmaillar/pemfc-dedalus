import pytest

from pemfc_dedalus.gas_permeability import (
    BARRER_TO_MOL_M_PER_M2_S_PA,
    barrer_to_si_permeability,
    membrane_permeance_from_permeability,
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
