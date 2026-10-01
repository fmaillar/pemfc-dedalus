"""Literature-anchored nitrogen crossover closure for V11.

The closure uses Catalano et al. Nafion 117 nitrogen permeability data and the
V11 membrane state. No Ballard purge trajectory or polarization data are used
for fitting.

A single lumped membrane water content is used as a one-point reduction of the
through-plane hydration profile.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gas_permeability import (
    arrhenius_permeability,
    barrer_to_si_permeability,
    calibrated_exponential_permeability_multiplier,
    membrane_permeance_from_permeability,
    membrane_water_volume_fraction_from_partial_molar_volume,
)
from .membrane import membrane_water_content_from_activity
from .v11_materials import V11MEAReference


@dataclass(frozen=True)
class CatalanoNitrogenReference:
    """Nafion 117 N2 permeability reference from Catalano et al. (2012)."""

    dry_reference_permeability_barrer: float = 0.18
    reference_temperature_k: float = 308.15
    dry_activation_energy_j_mol: float = 49_600.0
    anchor_water_activity: float = 0.75
    anchor_permeability_factor: float = 95.0
    maximum_permeability_factor: float = 100.0
    water_partial_molar_volume_m3_mol: float = 17.0e-6


@dataclass(frozen=True)
class V11NitrogenCrossover:
    """Resolved membrane N2 crossover quantities."""

    water_volume_fraction: float
    dry_water_volume_fraction: float
    anchor_water_volume_fraction: float
    hydration_multiplier: float
    permeability_mol_m_per_m2_s_pa: float
    permeance_mol_m2_s_pa: float
    cathode_partial_pressure_pa: float
    anode_partial_pressure_pa: float
    pressure_difference_pa: float
    flux_mol_m2_s: float
    rate_mol_s: float


def _scalar_water_content(activity: float) -> float:
    return float(membrane_water_content_from_activity(activity))


def catalano_nitrogen_crossover(
    *,
    membrane_mean_water_content: float,
    temperature_k: float,
    cathode_nitrogen_partial_pressure_pa: float,
    anode_nitrogen_partial_pressure_pa: float,
    mea: V11MEAReference | None = None,
    reference: CatalanoNitrogenReference | None = None,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> V11NitrogenCrossover:
    """Return cathode-to-anode N2 crossover from literature permeability.

    The water-content dependence is the exponential Catalano trend anchored to
    the reported approximately 95-fold enhancement at water activity 0.75.
    Temperature scaling uses the reported dry-Nafion N2 activation energy.

    Negative pressure gradients are clipped to zero because this reduced V11
    closure represents cathode-to-anode crossover only.
    """
    if membrane_mean_water_content < 0.0:
        raise ValueError("membrane_mean_water_content must be non-negative")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if cathode_nitrogen_partial_pressure_pa < 0.0:
        raise ValueError(
            "cathode_nitrogen_partial_pressure_pa must be non-negative"
        )
    if anode_nitrogen_partial_pressure_pa < 0.0:
        raise ValueError(
            "anode_nitrogen_partial_pressure_pa must be non-negative"
        )

    material = V11MEAReference() if mea is None else mea
    ref = CatalanoNitrogenReference() if reference is None else reference

    lambda_dry = _scalar_water_content(0.0)
    lambda_anchor = _scalar_water_content(ref.anchor_water_activity)

    def water_fraction(water_content: float) -> float:
        return membrane_water_volume_fraction_from_partial_molar_volume(
            water_content,
            membrane_equivalent_weight_kg_mol=(
                material.membrane_equivalent_weight_kg_mol
            ),
            membrane_dry_density_kg_m3=material.membrane_dry_density_kg_m3,
            water_partial_molar_volume_m3_mol=(
                ref.water_partial_molar_volume_m3_mol
            ),
        )

    phi_water = water_fraction(membrane_mean_water_content)
    phi_dry = water_fraction(lambda_dry)
    phi_anchor = water_fraction(lambda_anchor)

    hydration_multiplier = calibrated_exponential_permeability_multiplier(
        phi_water,
        baseline_fraction=phi_dry,
        anchor_fraction=phi_anchor,
        anchor_factor=ref.anchor_permeability_factor,
        maximum_factor=ref.maximum_permeability_factor,
    )

    dry_reference_si = barrer_to_si_permeability(
        ref.dry_reference_permeability_barrer
    )
    dry_at_temperature = arrhenius_permeability(
        dry_reference_si,
        temperature_k,
        ref.reference_temperature_k,
        ref.dry_activation_energy_j_mol,
        gas_constant_j_mol_k,
    )
    permeability = dry_at_temperature * hydration_multiplier
    permeance = membrane_permeance_from_permeability(
        permeability,
        material.membrane_thickness_m,
    )

    pressure_difference = max(
        cathode_nitrogen_partial_pressure_pa
        - anode_nitrogen_partial_pressure_pa,
        0.0,
    )
    flux = permeance * pressure_difference
    rate = flux * material.active_cell_area_m2

    return V11NitrogenCrossover(
        water_volume_fraction=phi_water,
        dry_water_volume_fraction=phi_dry,
        anchor_water_volume_fraction=phi_anchor,
        hydration_multiplier=hydration_multiplier,
        permeability_mol_m_per_m2_s_pa=permeability,
        permeance_mol_m2_s_pa=permeance,
        cathode_partial_pressure_pa=cathode_nitrogen_partial_pressure_pa,
        anode_partial_pressure_pa=anode_nitrogen_partial_pressure_pa,
        pressure_difference_pa=pressure_difference,
        flux_mol_m2_s=flux,
        rate_mol_s=rate,
    )
