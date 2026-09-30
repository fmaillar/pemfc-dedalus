import numpy as np
import pytest

from scripts.run_v07_n2_catalano_motupally import make_tabulated_permeance_model


def test_tabulated_permeance_model_bilinear_interpolation():
    rh_axis = np.array([0.0, 0.5, 1.0])
    current_axis = np.array([100.0, 200.0, 300.0])
    table = np.empty((rh_axis.size, current_axis.size))

    for i, rh in enumerate(rh_axis):
        for j, current in enumerate(current_axis):
            table[i, j] = 2.0 * rh + 3.0 * current

    model = make_tabulated_permeance_model(
        rh_axis,
        current_axis,
        table,
    )

    assert model(0.25, 150.0) == pytest.approx(450.5)
    assert model(0.75, 250.0) == pytest.approx(751.5)


def test_tabulated_permeance_model_clamps_outside_grid():
    rh_axis = np.array([0.0, 1.0])
    current_axis = np.array([100.0, 200.0])
    table = np.array([[1.0, 2.0], [3.0, 4.0]])

    model = make_tabulated_permeance_model(
        rh_axis,
        current_axis,
        table,
    )

    assert model(-1.0, 50.0) == pytest.approx(1.0)
    assert model(2.0, 300.0) == pytest.approx(4.0)
