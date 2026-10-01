"""Tests for the V11 geometry-derived cathode heat-transfer closure."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_heat_transfer import (
    V11CathodeChannelGeometry,
    ballard_required_coolant_air_slpm,
    ballard_required_coolant_mass_flow_kg_s,
    cathode_air_outlet_temperature_ballard_v11,
    cathode_air_outlet_temperature_geometry_v11,
    trapezoidal_fully_developed_nusselt,
)


def test_published_channel_geometry_gives_expected_hydraulic_diameter() -> None:
    geometry = V11CathodeChannelGeometry()

    assert geometry.cross_section_m2 == pytest.approx(5.0e-6)
    assert geometry.hydraulic_diameter_m == pytest.approx(2.198e-3, rel=2.0e-3)
    assert geometry.convective_area_per_cell_m2 == pytest.approx(
        0.04368,
        rel=2.0e-3,
    )


def test_sadasivam_nusselt_matches_reported_aspect_ratio_one_value() -> None:
    assert trapezoidal_fully_developed_nusselt(1.0) == pytest.approx(
        2.9761,
        abs=1.0e-4,
    )


def test_geometry_closure_places_outlet_between_inlet_and_stack() -> None:
    result = cathode_air_outlet_temperature_geometry_v11(
        stack_temperature_k=313.15,
        inlet_temperature_k=293.15,
        stack_air_flow_slpm=216.1,
        n_cells=10,
    )

    assert 293.15 < result.outlet_temperature_k < 313.15
    assert result.ntu > 0.0
    assert 0.0 < result.effectiveness < 1.0


def test_lower_airflow_has_higher_heat_exchanger_effectiveness() -> None:
    low = cathode_air_outlet_temperature_geometry_v11(
        stack_temperature_k=313.15,
        inlet_temperature_k=293.15,
        stack_air_flow_slpm=100.0,
        n_cells=10,
    )
    high = cathode_air_outlet_temperature_geometry_v11(
        stack_temperature_k=313.15,
        inlet_temperature_k=293.15,
        stack_air_flow_slpm=300.0,
        n_cells=10,
    )

    assert low.ntu > high.ntu
    assert low.effectiveness > high.effectiveness
    assert low.outlet_temperature_k > high.outlet_temperature_k


def test_outlet_temperature_satisfies_ntu_relation() -> None:
    result = cathode_air_outlet_temperature_geometry_v11(
        stack_temperature_k=313.15,
        inlet_temperature_k=293.15,
        stack_air_flow_slpm=216.1,
        n_cells=10,
    )

    expected_effectiveness = (
        result.outlet_temperature_k - 293.15
    ) / (313.15 - 293.15)
    assert expected_effectiveness == pytest.approx(result.effectiveness)



def test_ballard_manual_cooling_example_matches_page_58() -> None:
    result = cathode_air_outlet_temperature_ballard_v11(
        stack_temperature_k=72.0 + 273.15,
        inlet_temperature_k=40.0 + 273.15,
        stack_air_flow_slpm=283.0,
        air_specific_heat_j_kg_k=1006.0,
    )

    assert result.heat_removed_w == pytest.approx(56.48, rel=5.0e-3)


def test_inverse_ballard_cooling_recovers_manual_example_flow() -> None:
    mass_flow = ballard_required_coolant_mass_flow_kg_s(
        heat_removed_w=56.48,
        stack_temperature_k=72.0 + 273.15,
        inlet_temperature_k=40.0 + 273.15,
        air_specific_heat_j_kg_k=1006.0,
    )
    airflow = ballard_required_coolant_air_slpm(
        heat_removed_w=56.48,
        stack_temperature_k=72.0 + 273.15,
        inlet_temperature_k=40.0 + 273.15,
        air_specific_heat_j_kg_k=1006.0,
    )

    assert mass_flow == pytest.approx(6.08e-3, rel=1.0e-2)
    assert airflow == pytest.approx(283.0, rel=1.0e-2)


def test_ballard_cooling_uses_same_air_stream_for_heat_removal() -> None:
    result = cathode_air_outlet_temperature_ballard_v11(
        stack_temperature_k=313.15,
        inlet_temperature_k=293.15,
        stack_air_flow_slpm=216.132,
    )

    assert result.heat_removed_w > 0.0
    assert 293.15 < result.outlet_temperature_k < 313.15
    assert 0.0 < result.effectiveness < 1.0
