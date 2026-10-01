"""Tests for literature-anchored V11 ORR kinetics."""

from __future__ import annotations

import math

import pytest

from pemfc_dedalus.v11_orr import (
    NeyerlinORRReference,
    catalyst_roughness_factor,
    cathodic_tafel_activation_loss_v,
    geometric_exchange_current_density_a_m2,
    neyerlin_orr_activation_state,
    neyerlin_specific_exchange_current_density_a_cm2_pt,
)


def test_reference_exchange_current_density_is_recovered_at_reference_state() -> None:
    ref = NeyerlinORRReference()

    value = neyerlin_specific_exchange_current_density_a_cm2_pt(
        temperature_k=ref.reference_temperature_k,
        oxygen_partial_pressure_pa=ref.reference_oxygen_pressure_pa,
        reference=ref,
    )

    assert value == pytest.approx(
        ref.reference_exchange_current_density_a_cm2_pt
    )


def test_specific_exchange_current_decreases_at_lower_temperature() -> None:
    ref = NeyerlinORRReference()

    cold = neyerlin_specific_exchange_current_density_a_cm2_pt(
        temperature_k=313.15,
        oxygen_partial_pressure_pa=ref.reference_oxygen_pressure_pa,
        reference=ref,
    )
    warm = neyerlin_specific_exchange_current_density_a_cm2_pt(
        temperature_k=353.0,
        oxygen_partial_pressure_pa=ref.reference_oxygen_pressure_pa,
        reference=ref,
    )

    assert cold < warm


def test_specific_exchange_current_follows_oxygen_reaction_order() -> None:
    ref = NeyerlinORRReference()
    half_pressure = neyerlin_specific_exchange_current_density_a_cm2_pt(
        temperature_k=ref.reference_temperature_k,
        oxygen_partial_pressure_pa=0.5 * ref.reference_oxygen_pressure_pa,
        reference=ref,
    )

    expected = (
        ref.reference_exchange_current_density_a_cm2_pt
        * 0.5**ref.oxygen_reaction_order
    )
    assert half_pressure == pytest.approx(expected)


def test_roughness_factor_unit_conversion() -> None:
    roughness = catalyst_roughness_factor(
        platinum_loading_mg_cm2_geo=0.4,
        ecsa_m2_pt_g_pt=50.0,
    )

    assert roughness == pytest.approx(200.0)


def test_geometric_exchange_current_scales_with_roughness() -> None:
    ref = NeyerlinORRReference()
    roughness, geometric = geometric_exchange_current_density_a_m2(
        temperature_k=ref.reference_temperature_k,
        oxygen_partial_pressure_pa=ref.reference_oxygen_pressure_pa,
        platinum_loading_mg_cm2_geo=0.4,
        ecsa_m2_pt_g_pt=50.0,
        reference=ref,
    )

    expected_a_cm2_geo = (
        ref.reference_exchange_current_density_a_cm2_pt * roughness
    )
    assert geometric == pytest.approx(expected_a_cm2_geo * 1.0e4)


def test_tafel_loss_matches_analytic_expression() -> None:
    current_density = 1800.0
    exchange = 0.02
    temperature = 313.15

    loss = cathodic_tafel_activation_loss_v(
        current_density_a_m2=current_density,
        exchange_current_density_a_m2=exchange,
        temperature_k=temperature,
        transfer_coefficient=1.0,
    )

    expected = (
        8.31446261815324
        * temperature
        / 96485.33212
        * math.log(current_density / exchange)
    )
    assert loss == pytest.approx(expected)


def test_activation_loss_increases_when_oxygen_pressure_falls() -> None:
    common = dict(
        current_density_a_m2=1800.0,
        temperature_k=313.15,
        platinum_loading_mg_cm2_geo=0.4,
        ecsa_m2_pt_g_pt=50.0,
    )
    high = neyerlin_orr_activation_state(
        oxygen_partial_pressure_pa=21_000.0,
        **common,
    )
    low = neyerlin_orr_activation_state(
        oxygen_partial_pressure_pa=10_500.0,
        **common,
    )

    assert low.activation_loss_v > high.activation_loss_v
