import math

import numpy as np

from pemfc_dedalus.ballard_1020acs import (
    Ballard1020ACSTechnologyReference,
    UserStackConfiguration,
)


def test_family_reference_does_not_assume_cell_count():
    tech = Ballard1020ACSTechnologyReference()
    assert np.isclose(tech.width_m, 0.351)
    assert np.isclose(tech.length_m, 0.103)
    assert tech.current_max_a == 75.0


def test_user_stack_is_200_w_and_provisionally_ten_cells():
    stack = UserStackConfiguration()
    assert stack.n_cells == 10
    assert np.isclose(stack.rated_power_w, 200.0)
    assert np.isclose(stack.nominal_power_per_cell_w, 20.0)


def test_stack_height_uses_cell_count_only_as_configuration():
    stack = UserStackConfiguration(n_cells=10)
    assert np.isclose(stack.nominal_stack_height_m, 0.1104)


def test_stack_power_scales_with_cell_count():
    stack = UserStackConfiguration(n_cells=10)
    assert np.isclose(stack.stack_power_w(29.0, 0.76), 220.4)



def test_bol_typical_voltage_interpolates_manual_curve():
    tech = Ballard1020ACSTechnologyReference()

    assert np.isclose(tech.bol_typical_cell_voltage_v(14.5), 0.80)
    assert np.isclose(tech.bol_typical_cell_voltage_v(21.75), 0.78)


def test_bol_typical_voltage_rejects_out_of_range_current():
    tech = Ballard1020ACSTechnologyReference()

    with np.testing.assert_raises(ValueError):
        tech.bol_typical_cell_voltage_v(-1.0)

def test_optimum_temperature_reference():
    tech = Ballard1020ACSTechnologyReference()
    assert np.isclose(tech.optimum_stack_temperature_c(65.3), 60.619, atol=0.1)


def test_standard_purge_period_at_65_3_a():
    stack = UserStackConfiguration()
    assert np.isclose(stack.purge_period_s(65.3), 2300.0 / 65.3)
    assert 35.0 < stack.purge_period_s(65.3) < 36.0


def test_dead_end_and_purge_reference_values():
    tech = Ballard1020ACSTechnologyReference()
    assert np.isclose(tech.h2_pressure_opt_barg, 0.36)
    assert np.isclose(tech.h2_dead_end_stoich, 1.0)
    assert np.isclose(tech.h2_stoich_including_purge, 1.07)
    assert np.isclose(tech.h2_utilization_with_standard_purge, 0.93)


def test_total_purge_volume_scales_with_user_stack():
    stack = UserStackConfiguration(n_cells=10)
    assert np.isclose(stack.purge_volume_m3(), 200e-6)


def test_air_and_h2_consumption_relations():
    stack = UserStackConfiguration()
    assert stack.stoichiometric_air_slpm(29.0) > stack.reacted_h2_slpm(29.0)
    assert stack.coolant_air_target_slpm(29.0) > stack.stoichiometric_air_slpm(29.0)
    assert math.isfinite(stack.purge_period_s(29.0))
    assert math.isinf(stack.purge_period_s(0.0))


def test_stack_mass_and_thermal_mass_scale_with_cell_count():
    stack = UserStackConfiguration(n_cells=10)

    assert np.isclose(stack.dry_mass_kg, 3.7)
    assert np.isclose(stack.thermal_mass_j_k, 1000.0)



def test_manual_implies_ten_ml_anode_volume_per_cell():
    tech = Ballard1020ACSTechnologyReference()
    stack = UserStackConfiguration(n_cells=10)

    assert np.isclose(tech.anode_gas_volume_per_cell_m3, 10e-6)
    assert np.isclose(stack.anode_gas_volume_m3(), 100e-6)


def test_manual_purge_flow_rates_scale_with_cell_count():
    stack = UserStackConfiguration(n_cells=10)

    assert np.isclose(stack.runtime_purge_rate_min_slpm(), 24.0)
    assert np.isclose(stack.startup_purge_rate_slpm(), 60.0)


def test_manual_anode_pressure_drop_relation():
    tech = Ballard1020ACSTechnologyReference()

    assert np.isclose(tech.anode_pressure_drop_mbar(2.4), 74.904)
    assert np.isclose(tech.anode_pressure_drop_mbar(6.0), 187.26)
