import pytest

from pemfc_dedalus.anode import (
    AnodeWaterState,
    advance_anode_water_state,
    advance_pressure_regulated_hydrogen,
    anode_total_gas_pressure_pa,
    hydrogen_consumption_molar_rate,
    hydrogen_moles_for_total_pressure,
    ideal_gas_partial_pressure_pa,
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



def test_hydrogen_consumption_obeys_faraday_law():
    rate = hydrogen_consumption_molar_rate(2.0, 100000.0)
    assert rate == pytest.approx(1.0e-5)


def test_ideal_gas_pressure_round_trip_for_hydrogen_inventory():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    target_pressure = 137325.0
    h2_mol = hydrogen_moles_for_total_pressure(
        target_pressure,
        water_vapor_mol=0.0,
        **kwargs,
    )
    assert ideal_gas_partial_pressure_pa(h2_mol, **kwargs) == pytest.approx(
        target_pressure
    )


def test_total_pressure_adds_hydrogen_and_water_partial_pressures():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    p_h2 = ideal_gas_partial_pressure_pa(1.0e-3, **kwargs)
    p_h2o = ideal_gas_partial_pressure_pa(1.0e-5, **kwargs)
    p_total = anode_total_gas_pressure_pa(1.0e-3, 1.0e-5, **kwargs)
    assert p_total == pytest.approx(p_h2 + p_h2o)


def test_pressure_regulator_replenishes_consumed_hydrogen():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    target_pressure = 137325.0
    h2_initial = hydrogen_moles_for_total_pressure(
        target_pressure,
        water_vapor_mol=0.0,
        **kwargs,
    )
    h2_new, inlet_rate, consumption_rate = advance_pressure_regulated_hydrogen(
        h2_initial,
        current_a=2.0,
        water_vapor_mol=0.0,
        dt_s=1.0,
        target_total_pressure_pa=target_pressure,
        faraday_c_mol=96485.33212,
        **kwargs,
    )
    assert inlet_rate == pytest.approx(consumption_rate)
    assert h2_new == pytest.approx(h2_initial)


def test_pressure_regulator_closes_when_water_already_overpressurizes_volume():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    target_pressure = 137325.0
    h2_initial = hydrogen_moles_for_total_pressure(
        target_pressure,
        water_vapor_mol=0.0,
        **kwargs,
    )
    water_mol = 2.0e-4
    h2_new, inlet_rate, _ = advance_pressure_regulated_hydrogen(
        h2_initial,
        current_a=0.0,
        water_vapor_mol=water_mol,
        dt_s=1.0,
        target_total_pressure_pa=target_pressure,
        faraday_c_mol=96485.33212,
        **kwargs,
    )
    assert inlet_rate == 0.0
    assert h2_new == pytest.approx(h2_initial)
    assert anode_total_gas_pressure_pa(h2_new, water_mol, **kwargs) > target_pressure
