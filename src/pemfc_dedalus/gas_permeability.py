"""Gas-permeability conversions used by the reduced PEMFC crossover models."""

from __future__ import annotations

import math

# 1 Barrer = 1e-10 cm^3(STP) cm / (cm^2 s cmHg).
# Converted with 22.414e-3 m^3/mol at STP and 1 cmHg = 1333.22 Pa.
BARRER_TO_MOL_M_PER_M2_S_PA = 3.348e-16


def barrer_to_si_permeability(permeability_barrer: float) -> float:
    """Convert permeability from Barrer to mol m / (m2 s Pa)."""
    if permeability_barrer < 0.0:
        raise ValueError("permeability_barrer must be non-negative")
    return permeability_barrer * BARRER_TO_MOL_M_PER_M2_S_PA


def membrane_permeance_from_permeability(
    permeability_mol_m_per_m2_s_pa: float,
    membrane_thickness_m: float,
) -> float:
    """Convert bulk membrane permeability to permeance [mol/(m2 s Pa)]."""
    if permeability_mol_m_per_m2_s_pa < 0.0:
        raise ValueError("permeability must be non-negative")
    if membrane_thickness_m <= 0.0:
        raise ValueError("membrane_thickness_m must be positive")
    return permeability_mol_m_per_m2_s_pa / membrane_thickness_m



def arrhenius_permeability(
    reference_permeability: float,
    temperature_k: float,
    reference_temperature_k: float,
    activation_energy_j_mol: float,
    gas_constant_j_mol_k: float,
) -> float:
    """Scale permeability with an Arrhenius temperature law."""
    if reference_permeability < 0.0:
        raise ValueError("reference_permeability must be non-negative")
    if temperature_k <= 0.0 or reference_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    if activation_energy_j_mol < 0.0:
        raise ValueError("activation_energy_j_mol must be non-negative")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    exponent = (
        -activation_energy_j_mol
        / gas_constant_j_mol_k
        * (1.0 / temperature_k - 1.0 / reference_temperature_k)
    )
    return reference_permeability * math.exp(exponent)


def humidity_permeability_multiplier(
    relative_humidity: float,
    *,
    maximum_factor: float = 100.0,
    reference_relative_humidity: float = 0.90,
    shape_exponent: float = 2.0,
) -> float:
    """Return a bounded empirical humidity multiplier.

    Literature supports a strongly nonlinear humidity enhancement reaching up
    to roughly 100x by high RH for PFSI membranes, but does not provide one
    universal closed-form law. This interpolation is therefore explicitly a
    screening closure between dry conditions and the reported high-RH bound.
    """
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be in [0, 1]")
    if maximum_factor < 1.0:
        raise ValueError("maximum_factor must be >= 1")
    if not 0.0 < reference_relative_humidity <= 1.0:
        raise ValueError("reference_relative_humidity must be in (0, 1]")
    if shape_exponent <= 0.0:
        raise ValueError("shape_exponent must be positive")

    scaled = min(relative_humidity / reference_relative_humidity, 1.0)
    return maximum_factor ** (scaled**shape_exponent)


def effective_membrane_relative_humidity(
    anode_relative_humidity: float,
    cathode_relative_humidity: float,
) -> float:
    """Return the first-order mean boundary RH used by the screening closure."""
    if not 0.0 <= anode_relative_humidity <= 1.0:
        raise ValueError("anode_relative_humidity must be in [0, 1]")
    if not 0.0 <= cathode_relative_humidity <= 1.0:
        raise ValueError("cathode_relative_humidity must be in [0, 1]")
    return 0.5 * (anode_relative_humidity + cathode_relative_humidity)



def state_dependent_permeability(
    dry_reference_permeability: float,
    *,
    temperature_k: float,
    reference_temperature_k: float,
    activation_energy_j_mol: float,
    gas_constant_j_mol_k: float,
    anode_relative_humidity: float,
    cathode_relative_humidity: float,
    maximum_humidity_factor: float = 100.0,
    humidity_reference_relative_humidity: float = 0.90,
    humidity_shape_exponent: float = 2.0,
) -> float:
    """Combine Arrhenius temperature scaling with the RH screening closure."""
    dry_at_temperature = arrhenius_permeability(
        dry_reference_permeability,
        temperature_k,
        reference_temperature_k,
        activation_energy_j_mol,
        gas_constant_j_mol_k,
    )
    effective_rh = effective_membrane_relative_humidity(
        anode_relative_humidity,
        cathode_relative_humidity,
    )
    humidity_factor = humidity_permeability_multiplier(
        effective_rh,
        maximum_factor=maximum_humidity_factor,
        reference_relative_humidity=humidity_reference_relative_humidity,
        shape_exponent=humidity_shape_exponent,
    )
    return dry_at_temperature * humidity_factor



def water_content_permeability_multiplier(
    water_content: float,
    *,
    dry_water_content: float,
    reference_water_content: float,
    maximum_factor: float = 100.0,
) -> float:
    """Return an exponential multiplier based on membrane water content.

    The closure maps the dry reference water content to 1x permeability and a
    chosen hydrated reference water content to maximum_factor. It is a
    screening proxy for the exponential permeability increase reported versus
    membrane water volume fraction.
    """
    if water_content < 0.0:
        raise ValueError("water_content must be non-negative")
    if dry_water_content < 0.0:
        raise ValueError("dry_water_content must be non-negative")
    if reference_water_content <= dry_water_content:
        raise ValueError(
            "reference_water_content must exceed dry_water_content"
        )
    if maximum_factor < 1.0:
        raise ValueError("maximum_factor must be >= 1")

    normalized = (
        (water_content - dry_water_content)
        / (reference_water_content - dry_water_content)
    )
    bounded = min(max(normalized, 0.0), 1.0)
    return maximum_factor**bounded


def water_content_dependent_permeability(
    dry_reference_permeability: float,
    *,
    temperature_k: float,
    reference_temperature_k: float,
    activation_energy_j_mol: float,
    gas_constant_j_mol_k: float,
    water_content: float,
    dry_water_content: float,
    reference_water_content: float,
    maximum_humidity_factor: float = 100.0,
) -> float:
    """Combine Arrhenius scaling with a membrane-water-content closure."""
    dry_at_temperature = arrhenius_permeability(
        dry_reference_permeability,
        temperature_k,
        reference_temperature_k,
        activation_energy_j_mol,
        gas_constant_j_mol_k,
    )
    water_factor = water_content_permeability_multiplier(
        water_content,
        dry_water_content=dry_water_content,
        reference_water_content=reference_water_content,
        maximum_factor=maximum_humidity_factor,
    )
    return dry_at_temperature * water_factor


def membrane_water_volume_fraction(
    water_content: float,
    *,
    membrane_equivalent_weight_kg_mol: float,
    membrane_dry_density_kg_m3: float,
    water_molar_mass_kg_mol: float = 0.01801528,
    water_density_kg_m3: float = 1000.0,
) -> float:
    """Convert lambda to an additive-volume water fraction."""
    if water_content < 0.0:
        raise ValueError("water_content must be non-negative")
    if membrane_equivalent_weight_kg_mol <= 0.0:
        raise ValueError("membrane equivalent weight must be positive")
    if membrane_dry_density_kg_m3 <= 0.0:
        raise ValueError("membrane dry density must be positive")
    if water_molar_mass_kg_mol <= 0.0:
        raise ValueError("water molar mass must be positive")
    if water_density_kg_m3 <= 0.0:
        raise ValueError("water density must be positive")

    water_volume = (
        water_content * water_molar_mass_kg_mol / water_density_kg_m3
    )
    polymer_volume = (
        membrane_equivalent_weight_kg_mol / membrane_dry_density_kg_m3
    )
    return water_volume / (water_volume + polymer_volume)


def water_volume_fraction_permeability_multiplier(
    water_volume_fraction: float,
    *,
    lower_fraction: float = 0.02,
    upper_fraction: float = 0.20,
    upper_factor: float = 100.0,
) -> float:
    """Map water volume fraction to an exponential permeability multiplier.

    Catalano et al. report an approximately exponential increase over roughly
    2% to 20% water volume fraction. This closure uses those bounds as a
    screening interval and maps the upper bound to the reported order-of-
    magnitude enhancement.
    """
    if not 0.0 <= water_volume_fraction <= 1.0:
        raise ValueError("water_volume_fraction must be in [0, 1]")
    if not 0.0 <= lower_fraction < upper_fraction <= 1.0:
        raise ValueError("invalid water-volume-fraction bounds")
    if upper_factor < 1.0:
        raise ValueError("upper_factor must be >= 1")

    normalized = (
        (water_volume_fraction - lower_fraction)
        / (upper_fraction - lower_fraction)
    )
    bounded = min(max(normalized, 0.0), 1.0)
    return upper_factor**bounded


def water_volume_fraction_dependent_permeability(
    dry_reference_permeability: float,
    *,
    temperature_k: float,
    reference_temperature_k: float,
    activation_energy_j_mol: float,
    gas_constant_j_mol_k: float,
    water_volume_fraction: float,
    lower_fraction: float = 0.02,
    upper_fraction: float = 0.20,
    upper_factor: float = 100.0,
) -> float:
    """Combine Arrhenius scaling with a water-volume-fraction closure."""
    dry_at_temperature = arrhenius_permeability(
        dry_reference_permeability,
        temperature_k,
        reference_temperature_k,
        activation_energy_j_mol,
        gas_constant_j_mol_k,
    )
    water_factor = water_volume_fraction_permeability_multiplier(
        water_volume_fraction,
        lower_fraction=lower_fraction,
        upper_fraction=upper_fraction,
        upper_factor=upper_factor,
    )
    return dry_at_temperature * water_factor
