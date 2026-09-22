import numpy as np

from pemfc_dedalus.parameters import CathodeParameters
from pemfc_dedalus.physics import (
    butler_volmer_orr_current_density,
    cathode_overpotential,
    oxygen_consumption_from_current,
)


def test_overpotential_definition():
    eta = cathode_overpotential(0.78, 0.0, 1.18)
    assert np.isclose(eta, -0.40)


def test_butler_volmer_zero_at_equilibrium():
    p = CathodeParameters()
    j = butler_volmer_orr_current_density(
        p.oxygen_inlet_concentration,
        p.equilibrium_potential,
        0.0,
        p,
    )
    assert np.isclose(j, 0.0, atol=1e-12)


def test_butler_volmer_is_positive_for_cathodic_operation():
    p = CathodeParameters()
    j = butler_volmer_orr_current_density(
        p.oxygen_inlet_concentration,
        p.cathode_solid_potential,
        p.membrane_proton_potential,
        p,
    )
    assert j > 0.0


def test_orr_scales_linearly_with_oxygen_for_gamma_one():
    p = CathodeParameters(oxygen_reaction_order=1.0)
    c = p.oxygen_inlet_concentration
    j1 = butler_volmer_orr_current_density(c, 0.9, 0.0, p)
    j2 = butler_volmer_orr_current_density(0.5 * c, 0.9, 0.0, p)
    assert np.isclose(j2 / j1, 0.5, rtol=1e-12)


def test_four_electrons_per_oxygen_molecule():
    p = CathodeParameters()
    j = 4.0 * p.faraday
    sink = oxygen_consumption_from_current(j, p.faraday)
    assert np.isclose(sink, 1.0)
