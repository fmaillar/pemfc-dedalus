import numpy as np
import pytest

from scripts.run_v08_motupally_temperature_lookup import (
    make_trilinear_permeance_model,
)


def test_temperature_lookup_interpolates_between_planes():
    temperature_axis = np.asarray([300.0, 310.0])
    relative_humidity_axis = np.asarray([0.0, 1.0])
    current_density_axis = np.asarray([1000.0, 2000.0])
    lookup_tables = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )

    model = make_trilinear_permeance_model(
        temperature_axis,
        relative_humidity_axis,
        current_density_axis,
        lookup_tables,
    )

    low = model(0.5, 1500.0, 300.0)
    high = model(0.5, 1500.0, 310.0)
    mid = model(0.5, 1500.0, 305.0)

    assert low == pytest.approx(2.5)
    assert high == pytest.approx(5.0)
    assert mid == pytest.approx(3.75)


def test_temperature_lookup_clips_outside_temperature_axis():
    temperature_axis = np.asarray([300.0, 310.0])
    relative_humidity_axis = np.asarray([0.0, 1.0])
    current_density_axis = np.asarray([1000.0, 2000.0])
    lookup_tables = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )

    model = make_trilinear_permeance_model(
        temperature_axis,
        relative_humidity_axis,
        current_density_axis,
        lookup_tables,
    )

    assert model(0.5, 1500.0, 290.0) == pytest.approx(2.5)
    assert model(0.5, 1500.0, 320.0) == pytest.approx(5.0)
