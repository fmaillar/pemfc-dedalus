import numpy as np
import pytest

from pemfc_dedalus.membrane import (
    electro_osmotic_drag_coefficient,
    electro_osmotic_lambda_velocity,
    membrane_area_specific_resistance,
    membrane_fixed_charge_concentration,
    membrane_proton_conductivity,
    membrane_water_content_from_activity,
    steady_membrane_water_profile,
    steady_membrane_water_profile_zero_anode_flux,
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



def test_steady_membrane_profile_reduces_to_linear_without_drag():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 9)
    lam = steady_membrane_water_profile(
        z,
        lambda_anode=1.0,
        lambda_cathode=5.0,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=0.0,
    )
    assert np.allclose(lam, np.linspace(1.0, 5.0, 9))


def test_positive_drag_biases_water_profile_toward_anode_value():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 9)
    linear = np.linspace(1.0, 5.0, 9)
    lam = steady_membrane_water_profile(
        z,
        lambda_anode=1.0,
        lambda_cathode=5.0,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=1.0e-6,
    )
    assert lam[0] == pytest.approx(1.0)
    assert lam[-1] == pytest.approx(5.0)
    assert lam[4] < linear[4]


def test_membrane_asr_is_positive_and_decreases_with_hydration():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 65)
    dry = np.full_like(z, 2.0)
    wet = np.full_like(z, 8.0)
    asr_dry = membrane_area_specific_resistance(
        z,
        dry,
        temperature_k=p.stack_temperature,
        conductivity_floor_s_m=p.membrane_conductivity_floor,
    )
    asr_wet = membrane_area_specific_resistance(
        z,
        wet,
        temperature_k=p.stack_temperature,
        conductivity_floor_s_m=p.membrane_conductivity_floor,
    )
    assert asr_dry > asr_wet > 0.0



def test_zero_anode_flux_profile_keeps_dry_feed_membrane_hydrated():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 65)
    lam_cathode = membrane_water_content_from_activity(0.5).item()
    profile = steady_membrane_water_profile_zero_anode_flux(
        z,
        lambda_cathode=lam_cathode,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=6.0e-7,
    )
    assert 0.0 < profile[0] < profile[-1]
    assert profile[-1] == pytest.approx(lam_cathode)
    assert profile[0] > 0.5 * lam_cathode


def test_zero_flux_profile_is_uniform_without_drag():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 33)
    profile = steady_membrane_water_profile_zero_anode_flux(
        z,
        lambda_cathode=4.0,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=0.0,
    )
    assert np.allclose(profile, 4.0)
