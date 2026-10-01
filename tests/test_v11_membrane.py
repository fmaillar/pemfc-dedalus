"""Tests for reduced V11 membrane-water transport."""

from __future__ import annotations

import pytest

from pemfc_dedalus.v11_materials import V11MEAReference
from pemfc_dedalus.v11_membrane import (
    mean_water_content_from_inventory,
    membrane_fixed_site_moles_per_cell,
    membrane_hydration_rhs,
    membrane_water_inventory,
    membrane_water_transport,
)


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



def test_membrane_inventory_round_trip_lambda_to_water_and_back() -> None:
    lambda_mean = 7.5

    inventory = membrane_water_inventory(mean_water_content=lambda_mean)
    recovered = mean_water_content_from_inventory(
        water_mol=inventory.water_mol,
    )

    assert inventory.fixed_site_mol > 0.0
    assert inventory.water_mol == pytest.approx(
        lambda_mean * inventory.fixed_site_mol
    )
    assert recovered == pytest.approx(lambda_mean)


def test_fixed_site_moles_follow_v11_geometry_and_material_properties() -> None:
    mea = V11MEAReference()
    expected = (
        mea.membrane_dry_density_kg_m3
        / mea.membrane_equivalent_weight_kg_mol
        * mea.active_cell_area_m2
        * mea.membrane_thickness_m
    )

    assert membrane_fixed_site_moles_per_cell() == pytest.approx(expected)


def test_equal_and_opposite_interface_fluxes_conserve_membrane_inventory() -> None:
    flux = 2.5e-4

    rhs = membrane_hydration_rhs(
        anode_interface_flux_into_membrane_mol_m2_s=flux,
        cathode_interface_flux_into_membrane_mol_m2_s=-flux,
    )

    assert rhs.net_storage_rate_mol_s == pytest.approx(0.0)
    assert rhs.mean_water_content_rate_s == pytest.approx(0.0)


def test_net_interface_absorption_increases_mean_hydration() -> None:
    rhs = membrane_hydration_rhs(
        anode_interface_flux_into_membrane_mol_m2_s=1.0e-4,
        cathode_interface_flux_into_membrane_mol_m2_s=2.0e-4,
    )

    assert rhs.net_storage_rate_mol_s > 0.0
    assert rhs.mean_water_content_rate_s > 0.0


def test_net_interface_desorption_decreases_mean_hydration() -> None:
    rhs = membrane_hydration_rhs(
        anode_interface_flux_into_membrane_mol_m2_s=-1.0e-4,
        cathode_interface_flux_into_membrane_mol_m2_s=-2.0e-4,
    )

    assert rhs.net_storage_rate_mol_s < 0.0
    assert rhs.mean_water_content_rate_s < 0.0
