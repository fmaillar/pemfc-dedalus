import pytest

from pemfc_dedalus.membrane import (
    membrane_proton_conductivity,
    membrane_water_content_from_activity,
)
from pemfc_dedalus.parameters import CathodeParameters


def test_v04_cathode_hydration_uses_v03_constitutive_relations():
    p = CathodeParameters()
    lambda_cl = membrane_water_content_from_activity(p.relative_humidity).item()
    sigma_hydrated = membrane_proton_conductivity(
        lambda_cl,
        p.stack_temperature,
    ).item()

    assert lambda_cl == pytest.approx(3.4855, rel=1e-4)
    assert sigma_hydrated > 0.0
    assert sigma_hydrated < p.sigma_m_cl


def test_v04_drier_cathode_reduces_proton_conductivity():
    p = CathodeParameters()
    lambda_dry = membrane_water_content_from_activity(0.25).item()
    lambda_wet = membrane_water_content_from_activity(0.75).item()
    sigma_dry = membrane_proton_conductivity(lambda_dry, p.stack_temperature).item()
    sigma_wet = membrane_proton_conductivity(lambda_wet, p.stack_temperature).item()

    assert lambda_dry < lambda_wet
    assert sigma_dry < sigma_wet
