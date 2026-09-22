from pemfc_dedalus.parameters import CathodeParameters


def test_geometry_and_effective_diffusivity():
    p = CathodeParameters()

    assert p.thickness_z == p.gdl_thickness + p.cl_thickness
    assert 0.0 < p.d_o2_cl < p.d_o2_gdl < p.d_o2_bulk
    assert p.oxygen_inlet_concentration > 0.0


def test_open_cathode_operating_parameters_are_physical():
    p = CathodeParameters()

    assert 0.0 < p.oxygen_mole_fraction < 1.0
    assert p.temperature > 273.15
    assert p.pressure > 0.0
    assert 0.0 < p.porosity_cl < p.porosity_gdl < 1.0


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
