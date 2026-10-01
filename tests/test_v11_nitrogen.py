"""Tests for the V11 literature-anchored nitrogen crossover closure."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_nitrogen import (
    CatalanoNitrogenReference,
    catalano_nitrogen_crossover,
)


def test_dry_reference_recovers_catalano_permeability_at_reference_temperature() -> None:
    ref = CatalanoNitrogenReference()
    result = catalano_nitrogen_crossover(
        membrane_mean_water_content=0.043,
        temperature_k=ref.reference_temperature_k,
        cathode_nitrogen_partial_pressure_pa=80_000.0,
        anode_nitrogen_partial_pressure_pa=0.0,
        reference=ref,
    )

    expected_si = ref.dry_reference_permeability_barrer * 3.348e-16
    assert result.permeability_mol_m_per_m2_s_pa == pytest.approx(
        expected_si,
        rel=5.0e-3,
    )


def test_hydration_increases_nitrogen_permeability() -> None:
    dry = catalano_nitrogen_crossover(
        membrane_mean_water_content=0.043,
        temperature_k=308.15,
        cathode_nitrogen_partial_pressure_pa=80_000.0,
        anode_nitrogen_partial_pressure_pa=0.0,
    )
    hydrated = catalano_nitrogen_crossover(
        membrane_mean_water_content=7.0,
        temperature_k=308.15,
        cathode_nitrogen_partial_pressure_pa=80_000.0,
        anode_nitrogen_partial_pressure_pa=0.0,
    )

    assert hydrated.hydration_multiplier > dry.hydration_multiplier
    assert hydrated.permeability_mol_m_per_m2_s_pa > (
        dry.permeability_mol_m_per_m2_s_pa
    )


def test_nitrogen_crossover_is_pressure_driven_and_one_way() -> None:
    forward = catalano_nitrogen_crossover(
        membrane_mean_water_content=5.0,
        temperature_k=313.15,
        cathode_nitrogen_partial_pressure_pa=80_000.0,
        anode_nitrogen_partial_pressure_pa=5_000.0,
    )
    reversed_gradient = catalano_nitrogen_crossover(
        membrane_mean_water_content=5.0,
        temperature_k=313.15,
        cathode_nitrogen_partial_pressure_pa=5_000.0,
        anode_nitrogen_partial_pressure_pa=80_000.0,
    )

    assert forward.pressure_difference_pa == pytest.approx(75_000.0)
    assert forward.flux_mol_m2_s > 0.0
    assert forward.rate_mol_s > 0.0
    assert reversed_gradient.pressure_difference_pa == pytest.approx(0.0)
    assert reversed_gradient.rate_mol_s == pytest.approx(0.0)


def test_crossover_rate_is_flux_times_v11_active_area() -> None:
    result = catalano_nitrogen_crossover(
        membrane_mean_water_content=5.0,
        temperature_k=313.15,
        cathode_nitrogen_partial_pressure_pa=80_000.0,
        anode_nitrogen_partial_pressure_pa=5_000.0,
    )

    assert result.rate_mol_s == pytest.approx(result.flux_mol_m2_s * 0.0145)
