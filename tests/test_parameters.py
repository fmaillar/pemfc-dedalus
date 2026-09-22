from pemfc_dedalus.parameters import CathodeParameters


def test_geometry_and_effective_diffusivity():
    p = CathodeParameters()

    assert p.thickness_z == p.gdl_thickness + p.cl_thickness
    assert 0.0 < p.d_o2_cl < p.d_o2_gdl < p.d_o2_bulk
    assert p.oxygen_inlet_concentration > 0.0


def test_open_cathode_operating_parameters_are_physical():
    p = CathodeParameters()

    assert p.stack.n_cells == 10
    assert p.stack.rated_power_w == 200.0
    assert 0.0 < p.oxygen_mole_fraction < 1.0
    assert p.pressure == 101325.0
    assert 0.0 <= p.relative_humidity <= 1.0
    inlet_temp_c = p.oxidant_inlet_temperature - 273.15
    assert p.tech.oxidant_temp_min_c <= inlet_temp_c <= p.tech.oxidant_temp_max_c
    assert abs((p.stack_temperature - 273.15) - p.target_stack_temperature_c) < 1e-6
    assert 0.0 < p.porosity_cl < p.porosity_gdl < 1.0


def test_manual_derived_operating_targets():
    p = CathodeParameters()

    assert p.target_stack_temperature_c > 0.0
    assert p.target_air_flow_slpm > p.stack.stoichiometric_air_slpm(p.stack_current_a)
    assert p.purge_period_s == p.tech.purge_interval_as / p.stack_current_a
    assert p.purge_volume_m3 == p.stack.n_cells * p.tech.purge_volume_per_cell_m3


def test_electrochemistry_parameters_are_positive():
    p = CathodeParameters()

    assert p.faraday > 0.0
    assert p.gas_constant > 0.0
    assert p.j0_vol > 0.0
    assert p.sigma_s_gdl > 0.0
    assert p.sigma_s_cl > 0.0
    assert p.sigma_m_cl > 0.0
    assert p.sigma_m_floor > 0.0
    assert p.beta_anodic > 0.0
    assert p.beta_cathodic > 0.0
