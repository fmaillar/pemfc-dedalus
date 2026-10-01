"""Tests for reduced V11 membrane-water transport."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_membrane import membrane_water_transport
from pemfc_dedalus.v11_materials import V11MEAReference


def test_equal_hydration_eliminates_back_diffusion() -> None:
    result = membrane_water_transport(
        current_a=26.04,
        anode_water_activity=0.5,
        cathode_water_activity=0.5,
        temperature_k=313.15,
    )

    assert result.diffusive_flux_mol_m2_s == pytest.approx(0.0)
    assert result.electro_osmotic_flux_mol_m2_s > 0.0
    assert result.net_flux_mol_m2_s > 0.0


def test_wetter_cathode_drives_back_diffusion_to_anode() -> None:
    result = membrane_water_transport(
        current_a=0.0,
        anode_water_activity=0.2,
        cathode_water_activity=0.8,
        temperature_k=313.15,
    )

    assert result.lambda_cathode > result.lambda_anode
    assert result.electro_osmotic_flux_mol_m2_s == pytest.approx(0.0)
    assert result.diffusive_flux_mol_m2_s < 0.0
    assert result.net_rate_mol_s < 0.0


def test_electro_osmotic_rate_reduces_to_nd_times_i_over_f() -> None:
    current_a = 26.04
    result = membrane_water_transport(
        current_a=current_a,
        anode_water_activity=0.5,
        cathode_water_activity=0.5,
        temperature_k=313.15,
    )
    mea = V11MEAReference()
    electro_osmotic_rate = (
        result.electro_osmotic_flux_mol_m2_s * mea.active_cell_area_m2
    )
    expected = result.lambda_mean / 22.0 * current_a / 96485.33212

    assert electro_osmotic_rate == pytest.approx(expected)


def test_back_diffusion_grows_with_boundary_hydration_difference() -> None:
    weak = membrane_water_transport(
        current_a=0.0,
        anode_water_activity=0.4,
        cathode_water_activity=0.6,
        temperature_k=313.15,
    )
    strong = membrane_water_transport(
        current_a=0.0,
        anode_water_activity=0.2,
        cathode_water_activity=0.8,
        temperature_k=313.15,
    )

    assert abs(strong.diffusive_flux_mol_m2_s) > abs(
        weak.diffusive_flux_mol_m2_s
    )


def test_transport_uses_v11_membrane_geometry() -> None:
    mea = V11MEAReference(membrane_thickness_m=160e-6)
    thick = membrane_water_transport(
        current_a=0.0,
        anode_water_activity=0.2,
        cathode_water_activity=0.8,
        temperature_k=313.15,
        mea=mea,
    )
    reference = membrane_water_transport(
        current_a=0.0,
        anode_water_activity=0.2,
        cathode_water_activity=0.8,
        temperature_k=313.15,
    )

    assert thick.diffusive_flux_mol_m2_s == pytest.approx(
        0.5 * reference.diffusive_flux_mol_m2_s
    )
