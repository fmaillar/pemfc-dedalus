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
