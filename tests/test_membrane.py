import numpy as np
import pytest

from pemfc_dedalus.membrane import (
    electro_osmotic_drag_coefficient,
    electro_osmotic_lambda_velocity,
    membrane_fixed_charge_concentration,
    membrane_proton_conductivity,
    membrane_water_content_from_activity,
)
from pemfc_dedalus.parameters import CathodeParameters


def test_water_content_increases_with_activity():
    values = membrane_water_content_from_activity(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.diff(values) > 0.0)
    assert values[0] == pytest.approx(0.043)
    assert values[-1] == pytest.approx(14.003)


def test_drag_coefficient_is_lambda_over_22():
    lam = np.array([0.0, 11.0, 22.0])
    assert np.allclose(electro_osmotic_drag_coefficient(lam), [0.0, 0.5, 1.0])


def test_proton_conductivity_increases_with_hydration():
    p = CathodeParameters()
    sigma = membrane_proton_conductivity(
        np.array([2.0, 6.0, 12.0]),
        p.stack_temperature,
    )
    assert np.all(sigma >= 0.0)
    assert np.all(np.diff(sigma) > 0.0)


def test_fixed_charge_concentration_and_drag_velocity_are_positive():
    p = CathodeParameters()
    c_fixed = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    velocity = electro_osmotic_lambda_velocity(
        p.membrane_current_density,
        p.faraday,
        c_fixed,
    )
    assert c_fixed > 0.0
    assert velocity > 0.0


def test_membrane_parameter_assumptions_are_physical():
    p = CathodeParameters()
    assert p.membrane_thickness > 0.0
    assert p.membrane_water_diffusivity > 0.0
    assert 0.0 <= p.anode_relative_humidity <= 1.0
    assert p.membrane_current_density > 0.0
