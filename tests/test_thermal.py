import pytest

from pemfc_dedalus.thermal import (
    air_mass_flow_kg_s,
    open_cathode_airflow_target,
    outlet_temperature_k,
    required_air_flow_slpm,
    slpm_to_m3_s,
)


def test_slpm_conversion():
    assert slpm_to_m3_s(60.0) == pytest.approx(1.0e-3)


def test_air_mass_flow_is_positive():
    assert air_mass_flow_kg_s(216.0) > 0.0


def test_outlet_temperature_satisfies_energy_balance():
    outlet = outlet_temperature_k(
        heat_rejection_w=100.0,
        inlet_temperature_k=293.15,
        air_flow_slpm=300.0,
    )
    assert outlet > 293.15


def test_required_flow_inverts_outlet_temperature():
    target = 313.15
    flow = required_air_flow_slpm(
        heat_rejection_w=120.0,
        inlet_temperature_k=293.15,
        maximum_outlet_temperature_k=target,
    )
    outlet = outlet_temperature_k(
        heat_rejection_w=120.0,
        inlet_temperature_k=293.15,
        air_flow_slpm=flow,
    )
    assert outlet == pytest.approx(target)


def test_required_flow_rejects_invalid_temperature_window():
    with pytest.raises(ValueError):
        required_air_flow_slpm(
            heat_rejection_w=100.0,
            inlet_temperature_k=293.15,
            maximum_outlet_temperature_k=293.15,
        )


def test_airflow_target_uses_stoichiometric_floor_when_it_is_larger():
    result = open_cathode_airflow_target(
        heat_rejection_w=100.0,
        inlet_temperature_k=283.15,
        target_stack_temperature_k=313.15,
        stoichiometric_floor_slpm=300.0,
    )

    assert result.status == "reachable"
    assert result.active_constraint == "stoichiometric_floor"
    assert result.target_air_flow_slpm == pytest.approx(300.0)


def test_airflow_target_uses_thermal_requirement_when_it_is_larger():
    result = open_cathode_airflow_target(
        heat_rejection_w=120.0,
        inlet_temperature_k=293.15,
        target_stack_temperature_k=313.15,
        stoichiometric_floor_slpm=100.0,
    )

    assert result.status == "reachable"
    assert result.active_constraint == "thermal"
    assert result.target_air_flow_slpm is not None
    assert result.target_air_flow_slpm > 100.0


def test_airflow_target_flags_unreachable_ambient_temperature():
    result = open_cathode_airflow_target(
        heat_rejection_w=50.0,
        inlet_temperature_k=303.15,
        target_stack_temperature_k=303.15,
        stoichiometric_floor_slpm=100.0,
    )

    assert result.status == "target_unreachable"
    assert result.active_constraint == "ambient_temperature"
    assert result.target_air_flow_slpm is None
