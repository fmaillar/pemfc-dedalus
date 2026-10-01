"""Predictive V11 cell-voltage closure.

This module combines thermodynamic reversible potential, literature-anchored
ORR activation kinetics, membrane ohmic loss and explicitly supplied additional
resolved losses.

Cathode Pt loading and ECSA are mandatory inputs because FCgen-1020ACS-specific
values have not been independently established. They are never inferred from
the Ballard polarization curve.
"""

from __future__ import annotations

from dataclasses import dataclass

from .v11_galvanostatic import (
    resolved_cell_voltage_v,
    reversible_cell_voltage_liquid_water_v,
)
from .v11_materials import V11MEAReference
from .v11_orr import ORRKineticState, neyerlin_orr_activation_state


@dataclass(frozen=True)
class V11VoltagePrediction:
    """Resolved per-cell voltage terms and ORR diagnostics."""

    reversible_v: float
    activation_loss_v: float
    membrane_ohmic_loss_v: float
    additional_resolved_loss_v: float
    cell_voltage_v: float
    current_density_a_m2: float
    orr: ORRKineticState


def predict_cell_voltage_v(
    *,
    current_a: float,
    temperature_k: float,
    hydrogen_partial_pressure_pa: float,
    oxygen_partial_pressure_pa: float,
    water_activity: float,
    membrane_mean_water_content: float,
    cathode_platinum_loading_mg_cm2_geo: float,
    cathode_ecsa_m2_pt_g_pt: float,
    membrane_conductivity_multiplier: float = 1.0,
    additional_resolved_loss_v: float = 0.0,
    mea: V11MEAReference | None = None,
) -> V11VoltagePrediction:
    """Predict V11 cell voltage from resolved physical contributions.

    The returned value does not include an empirical concentration-loss term.
    Any independently resolved additional loss must be supplied explicitly.
    """
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if hydrogen_partial_pressure_pa <= 0.0:
        raise ValueError("hydrogen_partial_pressure_pa must be positive")
    if oxygen_partial_pressure_pa <= 0.0:
        raise ValueError("oxygen_partial_pressure_pa must be positive")
    if not 0.0 < water_activity <= 1.0:
        raise ValueError("water_activity must be in (0, 1]")
    if membrane_mean_water_content < 0.0:
        raise ValueError("membrane_mean_water_content must be non-negative")
    if membrane_conductivity_multiplier <= 0.0:
        raise ValueError("membrane_conductivity_multiplier must be positive")
    if additional_resolved_loss_v < 0.0:
        raise ValueError("additional_resolved_loss_v must be non-negative")

    reference = V11MEAReference() if mea is None else mea
    current_density = reference.current_density_a_m2(current_a)

    reversible = reversible_cell_voltage_liquid_water_v(
        temperature_k=temperature_k,
        hydrogen_partial_pressure_pa=hydrogen_partial_pressure_pa,
        oxygen_partial_pressure_pa=oxygen_partial_pressure_pa,
        water_activity=water_activity,
    )

    if current_a == 0.0:
        orr = ORRKineticState(
            catalyst_roughness_factor_cm2_pt_per_cm2_geo=(
                10.0
                * cathode_platinum_loading_mg_cm2_geo
                * cathode_ecsa_m2_pt_g_pt
            ),
            exchange_current_density_a_m2_geo=0.0,
            activation_loss_v=0.0,
        )
        activation = 0.0
        membrane_ohmic = 0.0
    else:
        orr = neyerlin_orr_activation_state(
            current_density_a_m2=current_density,
            temperature_k=temperature_k,
            oxygen_partial_pressure_pa=oxygen_partial_pressure_pa,
            platinum_loading_mg_cm2_geo=(
                cathode_platinum_loading_mg_cm2_geo
            ),
            ecsa_m2_pt_g_pt=cathode_ecsa_m2_pt_g_pt,
        )
        activation = orr.activation_loss_v
        membrane_ohmic = (
            reference.membrane_ohmic_loss_v(
                current_a=current_a,
                water_content=membrane_mean_water_content,
                temperature_k=temperature_k,
            )
            / membrane_conductivity_multiplier
        )

    breakdown = resolved_cell_voltage_v(
        reversible_v=reversible,
        activation_loss_v=activation,
        membrane_ohmic_loss_v=membrane_ohmic,
        other_resolved_loss_v=additional_resolved_loss_v,
    )

    return V11VoltagePrediction(
        reversible_v=reversible,
        activation_loss_v=activation,
        membrane_ohmic_loss_v=membrane_ohmic,
        additional_resolved_loss_v=additional_resolved_loss_v,
        cell_voltage_v=breakdown.cell_voltage_v,
        current_density_a_m2=current_density,
        orr=orr,
    )
