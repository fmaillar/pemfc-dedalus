"""Reduced membrane-water transport closure for V11.

Positive water flux is defined from anode to cathode. Electro-osmotic drag
therefore contributes positively, while back diffusion from a wetter cathode
to a drier anode contributes negatively.

The reduced closure uses the Springer activity-to-water-content relation,
n_d = lambda/22, and Motupally's state-dependent Nafion water diffusivity.
No fitted transport coefficient is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

from .membrane import (
    electro_osmotic_drag_coefficient,
    membrane_fixed_charge_concentration,
    membrane_water_content_from_activity,
    membrane_water_diffusivity_motupally,
    nafion_water_interfacial_transfer_coefficient_ge,
)
from .v11_materials import V11MEAReference


@dataclass(frozen=True)
class MembraneWaterTransport:
    """Reduced through-plane membrane water transport."""

    lambda_anode: float
    lambda_cathode: float
    lambda_mean: float
    diffusivity_m2_s: float
    electro_osmotic_flux_mol_m2_s: float
    diffusive_flux_mol_m2_s: float
    net_flux_mol_m2_s: float
    net_rate_mol_s: float


def membrane_water_transport(
    *,
    current_a: float,
    anode_water_activity: float,
    cathode_water_activity: float,
    temperature_k: float,
    mea: V11MEAReference | None = None,
    faraday_c_mol: float = 96485.33212,
) -> MembraneWaterTransport:
    """Return reduced membrane water flux, positive anode -> cathode.

    Boundary water contents are obtained from the Springer relation. Motupally
    diffusivity is evaluated at the arithmetic mean membrane water content,
    which is the explicit one-point reduction used by this lumped closure.
    """
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if not 0.0 <= anode_water_activity <= 1.0:
        raise ValueError("anode_water_activity must be in [0, 1]")
    if not 0.0 <= cathode_water_activity <= 1.0:
        raise ValueError("cathode_water_activity must be in [0, 1]")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if faraday_c_mol <= 0.0:
        raise ValueError("faraday_c_mol must be positive")

    reference = V11MEAReference() if mea is None else mea

    lambda_anode = float(
        membrane_water_content_from_activity(anode_water_activity).item()
    )
    lambda_cathode = float(
        membrane_water_content_from_activity(cathode_water_activity).item()
    )
    lambda_mean = 0.5 * (lambda_anode + lambda_cathode)

    diffusivity = float(
        membrane_water_diffusivity_motupally(
            lambda_mean,
            temperature_k,
        ).item()
    )
    fixed_charge = membrane_fixed_charge_concentration(
        reference.membrane_dry_density_kg_m3,
        reference.membrane_equivalent_weight_kg_mol,
    )
    current_density = reference.current_density_a_m2(current_a)
    drag_coefficient = float(
        electro_osmotic_drag_coefficient(lambda_mean).item()
    )

    electro_osmotic_flux = (
        drag_coefficient * current_density / faraday_c_mol
    )
    diffusive_flux = (
        -diffusivity
        * fixed_charge
        * (lambda_cathode - lambda_anode)
        / reference.membrane_thickness_m
    )
    net_flux = electro_osmotic_flux + diffusive_flux

    return MembraneWaterTransport(
        lambda_anode=lambda_anode,
        lambda_cathode=lambda_cathode,
        lambda_mean=lambda_mean,
        diffusivity_m2_s=diffusivity,
        electro_osmotic_flux_mol_m2_s=electro_osmotic_flux,
        diffusive_flux_mol_m2_s=diffusive_flux,
        net_flux_mol_m2_s=net_flux,
        net_rate_mol_s=net_flux * reference.active_cell_area_m2,
    )



@dataclass(frozen=True)
class MembraneWaterInventory:
    """Per-cell lumped membrane water inventory."""

    fixed_site_mol: float
    water_mol: float
    mean_water_content: float


@dataclass(frozen=True)
class MembraneHydrationDerivative:
    """Mean membrane hydration derivative from interfacial water exchange."""

    anode_interface_rate_mol_s: float
    cathode_interface_rate_mol_s: float
    net_storage_rate_mol_s: float
    mean_water_content_rate_s: float


def membrane_fixed_site_moles_per_cell(
    *,
    mea: V11MEAReference | None = None,
) -> float:
    """Return fixed-acid-site moles in one cell membrane.

    N_sites = (rho_dry / EW) * A_cell * L_mem.
    """
    reference = V11MEAReference() if mea is None else mea
    fixed_charge = membrane_fixed_charge_concentration(
        reference.membrane_dry_density_kg_m3,
        reference.membrane_equivalent_weight_kg_mol,
    )
    membrane_volume = (
        reference.active_cell_area_m2 * reference.membrane_thickness_m
    )
    return fixed_charge * membrane_volume


def membrane_water_inventory(
    *,
    mean_water_content: float,
    mea: V11MEAReference | None = None,
) -> MembraneWaterInventory:
    """Return membrane water inventory for a lumped mean lambda state."""
    if mean_water_content < 0.0:
        raise ValueError("mean_water_content must be non-negative")

    fixed_sites = membrane_fixed_site_moles_per_cell(mea=mea)
    return MembraneWaterInventory(
        fixed_site_mol=fixed_sites,
        water_mol=mean_water_content * fixed_sites,
        mean_water_content=mean_water_content,
    )


def mean_water_content_from_inventory(
    *,
    water_mol: float,
    mea: V11MEAReference | None = None,
) -> float:
    """Convert membrane water inventory [mol] to mean lambda."""
    if water_mol < 0.0:
        raise ValueError("water_mol must be non-negative")

    fixed_sites = membrane_fixed_site_moles_per_cell(mea=mea)
    return water_mol / fixed_sites


def membrane_hydration_rhs(
    *,
    anode_interface_flux_into_membrane_mol_m2_s: float,
    cathode_interface_flux_into_membrane_mol_m2_s: float,
    mea: V11MEAReference | None = None,
) -> MembraneHydrationDerivative:
    """Return d(lambda_mean)/dt from interfacial water exchange.

    Both interface flux arguments are positive when water enters the membrane
    from the adjacent phase. Through-plane EOD and back diffusion do not appear
    in this total-inventory balance because they redistribute water internally;
    only net exchange through the two external membrane interfaces changes the
    total membrane water content.
    """
    reference = V11MEAReference() if mea is None else mea
    anode_rate = (
        anode_interface_flux_into_membrane_mol_m2_s
        * reference.active_cell_area_m2
    )
    cathode_rate = (
        cathode_interface_flux_into_membrane_mol_m2_s
        * reference.active_cell_area_m2
    )
    net_rate = anode_rate + cathode_rate
    fixed_sites = membrane_fixed_site_moles_per_cell(mea=reference)

    return MembraneHydrationDerivative(
        anode_interface_rate_mol_s=anode_rate,
        cathode_interface_rate_mol_s=cathode_rate,
        net_storage_rate_mol_s=net_rate,
        mean_water_content_rate_s=net_rate / fixed_sites,
    )



@dataclass(frozen=True)
class MembraneInterfaceWaterFlux:
    """One gas/membrane interfacial water-transfer state."""

    mode: str
    equilibrium_water_content: float
    membrane_water_content: float
    water_volume_fraction: float
    transfer_coefficient_m_s: float
    flux_into_membrane_mol_m2_s: float


def membrane_water_volume_fraction(
    *,
    water_content: float,
    mea: V11MEAReference | None = None,
    water_molar_volume_m3_mol: float = 18.0e-6,
) -> float:
    """Return hydrated-ionomer water volume fraction.

    Uses the standard additive-volume form

        f_w = lambda V_w / (lambda V_w + V_m),

    with V_m = EW / rho_dry.
    """
    if water_content < 0.0:
        raise ValueError("water_content must be non-negative")
    if water_molar_volume_m3_mol <= 0.0:
        raise ValueError("water_molar_volume_m3_mol must be positive")

    reference = V11MEAReference() if mea is None else mea
    dry_membrane_molar_volume = (
        reference.membrane_equivalent_weight_kg_mol
        / reference.membrane_dry_density_kg_m3
    )
    water_volume = water_content * water_molar_volume_m3_mol
    return water_volume / (water_volume + dry_membrane_molar_volume)


def ge_interface_water_flux_into_membrane(
    *,
    gas_water_activity: float,
    membrane_water_content: float,
    temperature_k: float,
    mea: V11MEAReference | None = None,
    water_molar_volume_m3_mol: float = 18.0e-6,
) -> MembraneInterfaceWaterFlux:
    """Return Ge interfacial water flux, positive gas -> membrane.

    The equilibrium membrane hydration is obtained from the Springer isotherm.
    Absorption is selected when lambda_eq > lambda_mem and desorption otherwise.
    """
    if not 0.0 <= gas_water_activity <= 1.0:
        raise ValueError("gas_water_activity must be in [0, 1]")
    if membrane_water_content < 0.0:
        raise ValueError("membrane_water_content must be non-negative")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")

    reference = V11MEAReference() if mea is None else mea
    equilibrium = float(
        membrane_water_content_from_activity(gas_water_activity).item()
    )
    water_fraction = membrane_water_volume_fraction(
        water_content=membrane_water_content,
        mea=reference,
        water_molar_volume_m3_mol=water_molar_volume_m3_mol,
    )

    mode = "absorption" if equilibrium >= membrane_water_content else "desorption"
    coefficient = float(
        nafion_water_interfacial_transfer_coefficient_ge(
            water_fraction,
            mode=mode,
            temperature_k=temperature_k,
        ).item()
    )
    fixed_charge = membrane_fixed_charge_concentration(
        reference.membrane_dry_density_kg_m3,
        reference.membrane_equivalent_weight_kg_mol,
    )
    flux = coefficient * fixed_charge * (
        equilibrium - membrane_water_content
    )

    return MembraneInterfaceWaterFlux(
        mode=mode,
        equilibrium_water_content=equilibrium,
        membrane_water_content=membrane_water_content,
        water_volume_fraction=water_fraction,
        transfer_coefficient_m_s=coefficient,
        flux_into_membrane_mol_m2_s=flux,
    )


def membrane_hydration_rhs_ge(
    *,
    mean_water_content: float,
    anode_water_activity: float,
    cathode_water_activity: float,
    temperature_k: float,
    mea: V11MEAReference | None = None,
) -> MembraneHydrationDerivative:
    """Close the lumped membrane inventory with Ge interface kinetics."""
    reference = V11MEAReference() if mea is None else mea
    anode = ge_interface_water_flux_into_membrane(
        gas_water_activity=anode_water_activity,
        membrane_water_content=mean_water_content,
        temperature_k=temperature_k,
        mea=reference,
    )
    cathode = ge_interface_water_flux_into_membrane(
        gas_water_activity=cathode_water_activity,
        membrane_water_content=mean_water_content,
        temperature_k=temperature_k,
        mea=reference,
    )
    return membrane_hydration_rhs(
        anode_interface_flux_into_membrane_mol_m2_s=(
            anode.flux_into_membrane_mol_m2_s
        ),
        cathode_interface_flux_into_membrane_mol_m2_s=(
            cathode.flux_into_membrane_mol_m2_s
        ),
        mea=reference,
    )
