from pemfc_dedalus.gas_permeability import (
    barrer_to_si_permeability,
    membrane_permeance_from_permeability,
)


def test_literature_anchor_maps_to_expected_initial_flux():
    permeability = barrer_to_si_permeability(0.24)
    permeance = membrane_permeance_from_permeability(
        permeability,
        50.0e-6,
    )
    initial_flux = permeance * 77208.78477076966

    assert 1.0e-7 < initial_flux < 2.0e-7


def test_hundredfold_humidity_screening_scales_flux_hundredfold():
    dry = barrer_to_si_permeability(0.24)
    humid = barrer_to_si_permeability(24.0)

    assert humid == 100.0 * dry
