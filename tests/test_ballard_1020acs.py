import math

import numpy as np

from pemfc_dedalus.ballard_1020acs import Ballard1020ACSReference


def test_stack_geometry_reference():
    r = Ballard1020ACSReference()
    assert r.n_cells == 56
    assert np.isclose(r.width_m, 0.351)
    assert np.isclose(r.length_m, 0.103)
    assert np.isclose(r.stack_height_m, 0.3634)


def test_nominal_bol_point_is_kw_class():
    r = Ballard1020ACSReference()
    p = r.n_cells * 65.3 * 0.66
    assert np.isclose(p, 2413.488)
    assert p > 2000.0


def test_optimum_temperature_reference():
    r = Ballard1020ACSReference()
    assert np.isclose(r.optimum_stack_temperature_c(65.3), 60.619, atol=0.1)


def test_standard_purge_period_at_65_3_a():
    r = Ballard1020ACSReference()
    assert np.isclose(r.purge_period_s(65.3), 2300.0 / 65.3)
    assert 35.0 < r.purge_period_s(65.3) < 36.0


def test_dead_end_and_purge_reference_values():
    r = Ballard1020ACSReference()
    assert np.isclose(r.h2_pressure_opt_barg, 0.36)
    assert np.isclose(r.h2_dead_end_stoich, 1.0)
    assert np.isclose(r.h2_stoich_including_purge, 1.07)
    assert np.isclose(r.h2_utilization_with_standard_purge, 0.93)


def test_air_and_h2_consumption_relations():
    r = Ballard1020ACSReference()
    assert r.stoichiometric_air_slpm(65.3) > r.reacted_h2_slpm(65.3)
    assert math.isfinite(r.purge_period_s(65.3))
    assert math.isinf(r.purge_period_s(0.0))
