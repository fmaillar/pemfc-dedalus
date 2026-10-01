"""Tests for literature-sourced V11 MEA parameters."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_materials import V11MEAReference


def test_v11_mea_reference_uses_literature_geometry() -> None:
    ref = V11MEAReference()

    assert ref.active_cell_area_m2 == pytest.approx(0.0145)
    assert ref.membrane_thickness_m == pytest.approx(80e-6)
    assert ref.gdl_thickness_m == pytest.approx(200e-6)
    assert ref.catalyst_layer_thickness_m == pytest.approx(10e-6)
    assert ref.gdl_porosity == pytest.approx(0.76)
    assert ref.catalyst_layer_porosity == pytest.approx(0.40)


def test_v11_current_density_uses_reported_active_area() -> None:
    ref = V11MEAReference()

    assert ref.current_density_a_m2(26.04) == pytest.approx(
        1795.8620689655172
    )


def test_v11_membrane_asr_and_loss_are_positive() -> None:
    ref = V11MEAReference()

    conductivity = ref.membrane_proton_conductivity_s_m(
        water_content=6.0,
        temperature_k=313.15,
    )
    asr = ref.membrane_area_specific_resistance_ohm_m2(
        water_content=6.0,
        temperature_k=313.15,
    )
    loss = ref.membrane_ohmic_loss_v(
        current_a=26.04,
        water_content=6.0,
        temperature_k=313.15,
    )

    assert conductivity > 0.0
    assert asr > 0.0
    assert loss > 0.0
    assert asr == pytest.approx(ref.membrane_thickness_m / conductivity)


def test_wetter_membrane_has_lower_ohmic_loss() -> None:
    ref = V11MEAReference()

    dry = ref.membrane_ohmic_loss_v(
        current_a=26.04,
        water_content=3.0,
        temperature_k=313.15,
    )
    wet = ref.membrane_ohmic_loss_v(
        current_a=26.04,
        water_content=8.0,
        temperature_k=313.15,
    )

    assert wet < dry
