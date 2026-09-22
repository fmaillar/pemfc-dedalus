from pemfc_dedalus.parameters import CathodeParameters


def test_geometry_and_effective_diffusivity():
    p = CathodeParameters()

    assert p.thickness_z == p.gdl_thickness + p.cl_thickness
    assert 0.0 < p.d_o2_cl < p.d_o2_gdl < p.d_o2_bulk
    assert p.oxygen_inlet_concentration > 0.0
