import pytest

from pemfc_dedalus.anode import (
    AnodeWaterState,
    advance_anode_water_state,
    lambda_flux_to_water_molar_rate,
    repartition_anode_water,
    saturated_water_vapor_moles,
    water_saturation_pressure_pa,
    water_vapor_moles_from_relative_humidity,
)


def test_water_saturation_pressure_is_physical_near_40c():
    p_sat = water_saturation_pressure_pa(313.15)
    assert 7.0e3 < p_sat < 8.0e3


def test_relative_humidity_round_trip_uses_saturated_inventory():
    volume = 20.0e-6
    temperature = 313.15
    gas_constant = 8.31446261815324
    n_sat = saturated_water_vapor_moles(
        volume_m3=volume,
        temperature_k=temperature,
        gas_constant_j_mol_k=gas_constant,
    )
    n_half = water_vapor_moles_from_relative_humidity(
        0.5,
        volume_m3=volume,
        temperature_k=temperature,
        gas_constant_j_mol_k=gas_constant,
    )
    assert n_half == pytest.approx(0.5 * n_sat)


def test_lambda_flux_conversion_preserves_sign():
    rate = lambda_flux_to_water_molar_rate(
        4.0e-6,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
    )
    assert rate == pytest.approx(1.44e-8)

    reverse = lambda_flux_to_water_molar_rate(
        -4.0e-6,
        membrane_area_m2=2.0e-6,
        fixed_charge_mol_m3=1800.0,
    )
    assert reverse == pytest.approx(-rate)


def test_repartition_caps_vapour_at_saturation_and_stores_liquid():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    n_sat = saturated_water_vapor_moles(**kwargs)
    state = repartition_anode_water(1.5 * n_sat, **kwargs)

    assert state.vapor_mol == pytest.approx(n_sat)
    assert state.liquid_mol == pytest.approx(0.5 * n_sat)
    assert state.relative_humidity == pytest.approx(1.0)


def test_positive_source_raises_anode_relative_humidity():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    initial = AnodeWaterState(vapor_mol=0.0, liquid_mol=0.0, relative_humidity=0.0)
    updated = advance_anode_water_state(
        initial,
        water_source_mol_s=1.0e-8,
        dt_s=10.0,
        **kwargs,
    )
    assert updated.relative_humidity > initial.relative_humidity
    assert updated.liquid_mol == 0.0


def test_negative_source_cannot_make_water_inventory_negative():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    initial = AnodeWaterState(vapor_mol=1.0e-8, liquid_mol=0.0, relative_humidity=0.0)
    updated = advance_anode_water_state(
        initial,
        water_source_mol_s=-1.0,
        dt_s=1.0,
        **kwargs,
    )
    assert updated.vapor_mol == 0.0
    assert updated.liquid_mol == 0.0
    assert updated.relative_humidity == 0.0
