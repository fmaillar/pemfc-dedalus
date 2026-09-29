import pytest

from pemfc_dedalus.anode import AnodeWaterState
from pemfc_dedalus.anode_nitrogen import (
    AnodeGasState,
    hydrogen_moles_for_pressure_with_nitrogen,
    nitrogen_crossover_molar_rate,
    nitrogen_permeance_from_reference_flux,
    nitrogen_pressure_driven_flux,
    remove_well_mixed_h2_n2_h2o,
    total_gas_pressure_with_nitrogen_pa,
)


def test_nitrogen_crossover_flux_scales_with_active_area():
    rate = nitrogen_crossover_molar_rate(2.0e-6, 0.025)
    assert rate == pytest.approx(5.0e-8)


def test_hydrogen_target_accounts_for_nitrogen_partial_pressure():
    kwargs = dict(
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )
    target_pressure = 137325.0
    h2 = hydrogen_moles_for_pressure_with_nitrogen(
        target_pressure,
        nitrogen_mol=1.0e-5,
        water_vapor_mol=2.0e-5,
        **kwargs,
    )
    pressure = total_gas_pressure_with_nitrogen_pa(
        h2,
        1.0e-5,
        2.0e-5,
        **kwargs,
    )
    assert pressure == pytest.approx(target_pressure)


def test_well_mixed_three_species_purge_preserves_composition():
    state = AnodeGasState(
        hydrogen_mol=7.0e-5,
        nitrogen_mol=1.0e-5,
        water=AnodeWaterState(
            vapor_mol=2.0e-5,
            liquid_mol=0.0,
            relative_humidity=0.5,
        ),
    )
    new_state, h2_out, n2_out, water_out = remove_well_mixed_h2_n2_h2o(
        state,
        gas_outflow_mol=1.0e-5,
        volume_m3=20.0e-6,
        temperature_k=313.15,
        gas_constant_j_mol_k=8.31446261815324,
    )

    assert h2_out == pytest.approx(7.0e-6)
    assert n2_out == pytest.approx(1.0e-6)
    assert water_out == pytest.approx(2.0e-6)
    assert new_state.hydrogen_mol == pytest.approx(6.3e-5)
    assert new_state.nitrogen_mol == pytest.approx(9.0e-6)
    assert new_state.water.vapor_mol == pytest.approx(1.8e-5)



def test_pressure_driven_nitrogen_flux_matches_reference_at_zero_anode_pressure():
    permeance = nitrogen_permeance_from_reference_flux(1.0e-6, 80000.0)
    flux = nitrogen_pressure_driven_flux(permeance, 80000.0, 0.0)
    assert flux == pytest.approx(1.0e-6)


def test_pressure_driven_nitrogen_flux_falls_with_anode_pressure():
    permeance = nitrogen_permeance_from_reference_flux(1.0e-6, 80000.0)
    assert nitrogen_pressure_driven_flux(
        permeance,
        80000.0,
        40000.0,
    ) == pytest.approx(5.0e-7)
    assert nitrogen_pressure_driven_flux(
        permeance,
        80000.0,
        90000.0,
    ) == 0.0
