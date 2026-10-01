import numpy as np
import pytest

from pemfc_dedalus.membrane import (
    anode_water_removal_flux_lambda_m_s,
    electro_osmotic_drag_coefficient,
    electro_osmotic_lambda_velocity,
    membrane_area_specific_resistance,
    membrane_fixed_charge_concentration,
    membrane_proton_conductivity,
    membrane_water_content_from_activity,
    membrane_water_content_grimaldi_da,
    membrane_water_diffusivity_grimaldi,
    membrane_water_diffusivity_motupally,
    nafion_water_interfacial_transfer_coefficient_ge,
    nafion_water_interfacial_transfer_coefficient_grimaldi,
    steady_membrane_water_profile,
    steady_membrane_water_profile_anode_transfer,
    steady_membrane_water_profile_grimaldi_consistent,
    steady_membrane_water_profile_motupally,
    steady_membrane_water_profile_motupally_grimaldi,
    steady_membrane_water_profile_two_interface_transfer,
    steady_membrane_water_profile_variable_diffusivity,
    steady_membrane_water_profile_variable_transfer,
    steady_membrane_water_profile_zero_anode_flux,
)
from pemfc_dedalus.parameters import CathodeParameters


def test_water_content_increases_with_activity():
    values = membrane_water_content_from_activity(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.diff(values) > 0.0)
    assert values[0] == pytest.approx(0.043)
    assert values[-1] == pytest.approx(14.003)




def test_ge_interfacial_transfer_correlation_matches_published_values():
    fractions = np.array([0.1, 0.2])

    absorption = nafion_water_interfacial_transfer_coefficient_ge(
        fractions,
        mode="absorption",
        temperature_k=303.0,
    )
    desorption = nafion_water_interfacial_transfer_coefficient_ge(
        fractions,
        mode="desorption",
        temperature_k=303.0,
    )

    assert np.allclose(absorption, 1.14e-5 * fractions)
    assert np.allclose(desorption, 4.59e-5 * fractions)
    assert np.all(desorption > absorption)


def test_ge_interfacial_transfer_correlation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        nafion_water_interfacial_transfer_coefficient_ge(
            -0.1,
            mode="absorption",
        )
    with pytest.raises(ValueError):
        nafion_water_interfacial_transfer_coefficient_ge(
            0.1,
            mode="invalid",
        )


def test_grimaldi_interfacial_transfer_correlation_matches_formula():
    water_content = np.array([2.0, 4.0])
    temperature = 313.15
    expected = (
        0.66e-6
        * water_content**1.6
        * np.exp(
            6000.0
            / 8.31446261815324
            * (1.0 / 303.0 - 1.0 / temperature)
        )
    )

    actual = nafion_water_interfacial_transfer_coefficient_grimaldi(
        water_content,
        temperature,
    )

    assert np.allclose(actual, expected)


def test_grimaldi_interfacial_transfer_increases_with_lambda_and_temperature():
    low_lambda = nafion_water_interfacial_transfer_coefficient_grimaldi(
        2.0,
        313.15,
    ).item()
    high_lambda = nafion_water_interfacial_transfer_coefficient_grimaldi(
        4.0,
        313.15,
    ).item()
    hotter = nafion_water_interfacial_transfer_coefficient_grimaldi(
        4.0,
        333.15,
    ).item()

    assert high_lambda > low_lambda > 0.0
    assert hotter > high_lambda


def test_grimaldi_da_isotherm_matches_published_form():
    temperature = 313.15
    rh = np.array([0.0, 0.5, 1.0])

    values = membrane_water_content_grimaldi_da(rh, temperature)

    adsorption_potential = -8.31446261815324 * temperature * np.log(0.5)
    expected_half = 15.01 * np.exp(
        -(adsorption_potential / 1047.0) ** 0.4712
    )

    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(expected_half)
    assert values[2] == pytest.approx(15.01)
    assert np.all(np.diff(values) > 0.0)


def test_grimaldi_diffusivity_matches_published_form():
    p = CathodeParameters()
    water_content = 4.0
    temperature = p.stack_temperature
    membrane_molar_volume = (
        p.membrane_equivalent_weight / p.membrane_dry_density
    )
    expected = (
        6.47e-6
        * (
            1.0
            + 1.8e-5 / membrane_molar_volume * water_content
        )
        ** -2.0
        * (1.0 + 2.7e-3 * water_content**2)
        * (1.0 + np.tanh((water_content - 2.15) / 0.8758))
        * np.exp(-27800.0 / (p.gas_constant * temperature))
    )

    actual = membrane_water_diffusivity_grimaldi(
        water_content,
        temperature,
        equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        dry_density_kg_m3=p.membrane_dry_density,
        gas_constant_j_mol_k=p.gas_constant,
    ).item()

    assert actual == pytest.approx(expected)
    assert actual > 0.0


@pytest.mark.slow
def test_two_interface_solver_recovers_symmetric_equilibrium_without_drag():
    z = np.linspace(0.0, 50.0e-6, 129)

    profile = steady_membrane_water_profile_two_interface_transfer(
        z,
        lambda_anode_equilibrium=4.0,
        lambda_cathode_equilibrium=4.0,
        drag_velocity_m_s=0.0,
        diffusivity_model=lambda _value: 2.0e-10,
        transfer_coefficient_model=lambda _value: 5.0e-6,
    )

    assert np.allclose(profile, 4.0, rtol=0.0, atol=2.0e-5)


@pytest.mark.slow
def test_grimaldi_consistent_profile_is_physical():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)

    profile = steady_membrane_water_profile_grimaldi_consistent(
        z,
        anode_relative_humidity=0.5,
        cathode_relative_humidity=p.relative_humidity,
        temperature_k=p.stack_temperature,
        drag_velocity_m_s=2.0e-7,
        equivalent_weight_kg_mol=p.membrane_equivalent_weight,
        dry_density_kg_m3=p.membrane_dry_density,
        gas_constant_j_mol_k=p.gas_constant,
    )

    assert np.all(np.isfinite(profile))
    assert np.all(profile > 0.0)
    assert np.all(profile < 17.0)

def test_motupally_water_diffusivity_matches_published_branches():
    temperature = 313.15
    low_lambda = 2.5
    high_lambda = 5.0

    low_expected = (
        1.0e-4
        * 3.10e-3
        * low_lambda
        * (np.exp(0.28 * low_lambda) - 1.0)
        * np.exp(-2436.0 / temperature)
    )
    high_expected = (
        1.0e-4
        * 4.17e-4
        * (1.0 + 161.0 * np.exp(-high_lambda))
        * np.exp(-2436.0 / temperature)
    )

    values = membrane_water_diffusivity_motupally(
        np.array([low_lambda, high_lambda]),
        temperature,
    )

    assert values[0] == pytest.approx(low_expected)
    assert values[1] == pytest.approx(high_expected)
    assert np.all(values > 0.0)


def test_motupally_water_diffusivity_retains_lambda_three_discontinuity():
    temperature = 313.15
    below = membrane_water_diffusivity_motupally(3.0, temperature).item()
    above = membrane_water_diffusivity_motupally(3.0001, temperature).item()

    assert below > above


def test_motupally_water_diffusivity_rejects_invalid_state():
    with pytest.raises(ValueError):
        membrane_water_diffusivity_motupally(0.0, 313.15)
    with pytest.raises(ValueError):
        membrane_water_diffusivity_motupally(17.0, 313.15)
    with pytest.raises(ValueError):
        membrane_water_diffusivity_motupally(5.0, 0.0)

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



@pytest.mark.slow
def test_variable_diffusivity_solver_recovers_constant_diffusivity_profile():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)
    lambda_cathode = 4.0
    lambda_equilibrium = 1.5
    drag_velocity = 4.0e-7
    diffusivity = 2.0e-10

    expected = steady_membrane_water_profile_anode_transfer(
        z,
        lambda_cathode=lambda_cathode,
        lambda_anode_equilibrium=lambda_equilibrium,
        diffusivity_m2_s=diffusivity,
        drag_velocity_m_s=drag_velocity,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )
    actual = steady_membrane_water_profile_variable_diffusivity(
        z,
        lambda_cathode=lambda_cathode,
        lambda_anode_equilibrium=lambda_equilibrium,
        drag_velocity_m_s=drag_velocity,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
        diffusivity_model=lambda _value: diffusivity,
    )

    assert np.allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)


@pytest.mark.slow
def test_motupally_profile_hits_cathode_boundary_and_stays_physical():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)

    profile = steady_membrane_water_profile_motupally(
        z,
        lambda_cathode=3.4855,
        lambda_anode_equilibrium=9.4936,
        temperature_k=p.stack_temperature,
        drag_velocity_m_s=3.0e-7,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )

    assert np.all(profile > 0.0)
    assert np.all(profile < 17.0)
    assert profile[-1] == pytest.approx(3.4855, abs=1.0e-6)
    assert profile[0] > profile[-1]


@pytest.mark.slow
def test_motupally_profile_handles_dry_anode_startup():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)
    lambda_cathode = membrane_water_content_from_activity(
        p.relative_humidity
    ).item()
    lambda_anode_equilibrium = membrane_water_content_from_activity(0.0).item()

    profile = steady_membrane_water_profile_motupally(
        z,
        lambda_cathode=lambda_cathode,
        lambda_anode_equilibrium=lambda_anode_equilibrium,
        temperature_k=p.stack_temperature,
        drag_velocity_m_s=1.0e-7,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )

    assert np.all(profile > 0.0)
    assert np.all(profile < 17.0)
    assert profile[-1] == pytest.approx(lambda_cathode, abs=1.0e-8)


@pytest.mark.slow
def test_motupally_profile_brackets_nominal_transition_state():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    drag_velocity = electro_osmotic_lambda_velocity(
        1400.0,
        p.faraday,
        fixed_charge,
    )

    profile = steady_membrane_water_profile_motupally(
        z,
        lambda_cathode=3.4855,
        lambda_anode_equilibrium=3.20825,
        temperature_k=p.stack_temperature,
        drag_velocity_m_s=drag_velocity,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )

    assert np.all(profile > 0.0)
    assert np.all(profile < 17.0)
    assert profile[-1] == pytest.approx(3.4855, abs=1.0e-8)


@pytest.mark.slow
def test_motupally_profile_handles_equal_anode_cathode_equilibrium():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)
    lambda_equilibrium = membrane_water_content_from_activity(0.5).item()
    fixed_charge = membrane_fixed_charge_concentration(
        p.membrane_dry_density,
        p.membrane_equivalent_weight,
    )
    drag_velocity = electro_osmotic_lambda_velocity(
        1400.0,
        p.faraday,
        fixed_charge,
    )

    profile = steady_membrane_water_profile_motupally(
        z,
        lambda_cathode=lambda_equilibrium,
        lambda_anode_equilibrium=lambda_equilibrium,
        temperature_k=p.stack_temperature,
        drag_velocity_m_s=drag_velocity,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )

    assert np.all(profile > 0.0)
    assert np.all(profile < 17.0)
    assert profile[-1] == pytest.approx(lambda_equilibrium, abs=1.0e-8)


@pytest.mark.slow
def test_variable_transfer_solver_recovers_constant_transfer_solution():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)
    lambda_cathode = 4.0
    lambda_equilibrium = 1.5
    drag_velocity = 4.0e-7
    diffusivity = 2.0e-10
    transfer = p.anode_water_transfer_coefficient

    expected = steady_membrane_water_profile_variable_diffusivity(
        z,
        lambda_cathode=lambda_cathode,
        lambda_anode_equilibrium=lambda_equilibrium,
        drag_velocity_m_s=drag_velocity,
        anode_transfer_coefficient_m_s=transfer,
        diffusivity_model=lambda _value: diffusivity,
    )
    actual = steady_membrane_water_profile_variable_transfer(
        z,
        lambda_cathode=lambda_cathode,
        lambda_anode_equilibrium=lambda_equilibrium,
        drag_velocity_m_s=drag_velocity,
        diffusivity_model=lambda _value: diffusivity,
        transfer_coefficient_model=lambda _value: transfer,
    )

    assert np.allclose(actual, expected, rtol=3.0e-4, atol=3.0e-6)


@pytest.mark.slow
def test_motupally_grimaldi_profile_hits_cathode_boundary():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)

    profile = steady_membrane_water_profile_motupally_grimaldi(
        z,
        lambda_cathode=3.4855,
        lambda_anode_equilibrium=5.37,
        temperature_k=p.stack_temperature,
        drag_velocity_m_s=3.0e-7,
        gas_constant_j_mol_k=p.gas_constant,
    )

    assert np.all(profile > 0.0)
    assert np.all(profile < 17.0)
    assert profile[-1] == pytest.approx(3.4855, abs=1.0e-8)


@pytest.mark.slow
def test_grimaldi_profile_handles_wet_high_lambda_state():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 129)

    profile = steady_membrane_water_profile_motupally_grimaldi(
        z,
        lambda_cathode=3.4855,
        lambda_anode_equilibrium=10.0,
        temperature_k=p.stack_temperature,
        drag_velocity_m_s=2.0e-7,
        gas_constant_j_mol_k=p.gas_constant,
    )

    assert np.all(profile > 0.0)
    assert np.all(profile < 17.0)
    assert profile[-1] == pytest.approx(3.4855, abs=1.0e-8)

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


def test_anode_transfer_zero_coefficient_recovers_v05_zero_flux():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 65)
    lam_cathode = membrane_water_content_from_activity(0.5).item()
    velocity = 6.0e-7

    v05 = steady_membrane_water_profile_zero_anode_flux(
        z,
        lambda_cathode=lam_cathode,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=velocity,
    )
    v06 = steady_membrane_water_profile_anode_transfer(
        z,
        lambda_cathode=lam_cathode,
        lambda_anode_equilibrium=membrane_water_content_from_activity(
            p.anode_relative_humidity
        ).item(),
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=velocity,
        anode_transfer_coefficient_m_s=0.0,
    )
    assert np.allclose(v06, v05)


def test_finite_anode_transfer_drains_membrane_toward_dry_feed():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 65)
    lam_cathode = membrane_water_content_from_activity(0.5).item()
    lam_equilibrium = membrane_water_content_from_activity(0.0).item()
    velocity = 6.0e-7

    zero_flux = steady_membrane_water_profile_zero_anode_flux(
        z,
        lambda_cathode=lam_cathode,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=velocity,
    )
    finite_transfer = steady_membrane_water_profile_anode_transfer(
        z,
        lambda_cathode=lam_cathode,
        lambda_anode_equilibrium=lam_equilibrium,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=velocity,
        anode_transfer_coefficient_m_s=p.anode_water_transfer_coefficient,
    )

    assert lam_equilibrium < finite_transfer[0] < zero_flux[0]
    assert finite_transfer[-1] == pytest.approx(lam_cathode)
    assert np.mean(finite_transfer) < np.mean(zero_flux)


def test_anode_transfer_zero_drag_has_expected_robin_limit():
    p = CathodeParameters()
    z = np.linspace(0.0, p.membrane_thickness, 33)
    lam_cathode = 4.0
    lam_equilibrium = 1.0
    k = p.anode_water_transfer_coefficient

    profile = steady_membrane_water_profile_anode_transfer(
        z,
        lambda_cathode=lam_cathode,
        lambda_anode_equilibrium=lam_equilibrium,
        diffusivity_m2_s=p.membrane_water_diffusivity,
        drag_velocity_m_s=0.0,
        anode_transfer_coefficient_m_s=k,
    )
    transfer_number = k * p.membrane_thickness / p.membrane_water_diffusivity
    expected_anode = (
        lam_cathode + transfer_number * lam_equilibrium
    ) / (1.0 + transfer_number)

    assert profile[0] == pytest.approx(expected_anode)
    assert profile[-1] == pytest.approx(lam_cathode)


def test_anode_water_removal_flux_is_positive_above_equilibrium():
    flux = anode_water_removal_flux_lambda_m_s(3.0, 1.0, 2.0e-6)
    assert flux == pytest.approx(4.0e-6)



def test_ge_interfacial_transfer_increases_with_temperature():
    fraction = 0.2
    cold = nafion_water_interfacial_transfer_coefficient_ge(
        fraction,
        mode="absorption",
        temperature_k=303.0,
    ).item()
    warm = nafion_water_interfacial_transfer_coefficient_ge(
        fraction,
        mode="absorption",
        temperature_k=313.15,
    ).item()

    assert warm > cold
