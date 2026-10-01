"""Tests for V0.9 finite-transfer thermal closure."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.thermal import (
    airflow_multiplier_for_effectiveness,
    finite_transfer_cooling_power_w,
    heat_exchanger_effectiveness_from_ntu,
    ideal_air_cooling_power_w,
)


def test_effectiveness_limits() -> None:
    assert heat_exchanger_effectiveness_from_ntu(0.0) == pytest.approx(0.0)
    assert heat_exchanger_effectiveness_from_ntu(1.0) == pytest.approx(
        1.0 - np.exp(-1.0)
    )
    assert heat_exchanger_effectiveness_from_ntu(20.0) == pytest.approx(
        1.0,
        rel=1.0e-8,
    )


def test_finite_transfer_tends_to_ideal_at_large_ntu() -> None:
    kwargs = {
        "stack_temperature_k": 313.15,
        "inlet_temperature_k": 293.15,
        "air_flow_slpm": 200.0,
    }
    ideal = ideal_air_cooling_power_w(**kwargs)
    finite = finite_transfer_cooling_power_w(**kwargs, ntu=20.0)
    assert finite == pytest.approx(ideal, rel=1.0e-8)


def test_airflow_multiplier_is_inverse_effectiveness() -> None:
    assert airflow_multiplier_for_effectiveness(0.5) == pytest.approx(2.0)
    assert airflow_multiplier_for_effectiveness(1.0) == pytest.approx(1.0)


def test_negative_ntu_is_rejected() -> None:
    with pytest.raises(ValueError, match="ntu must be non-negative"):
        heat_exchanger_effectiveness_from_ntu(-0.1)
