"""Literature-anchored ORR kinetics for V11.

The catalyst-specific Pt/C kinetics follow the zero-overpotential parameter set
reported by Neyerlin et al. The conversion to geometric exchange current
requires cathode Pt loading and electrochemically active Pt surface area
(ECSA); these remain explicit because FCgen-1020ACS-specific values have not
been independently established.

The kinetic law is used in its cathodic Tafel regime, consistent with the
parameter extraction basis. No Ballard polarization data are used to tune it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeyerlinORRReference:
    """Catalyst-specific Pt/C ORR kinetic reference."""

    reference_temperature_k: float = 353.0
    reference_oxygen_pressure_pa: float = 101300.0
    reference_exchange_current_density_a_cm2_pt: float = 2.5e-8
    oxygen_reaction_order: float = 0.54
    cathodic_transfer_coefficient: float = 1.0
    activation_energy_j_mol: float = 67_000.0


@dataclass(frozen=True)
class ORRKineticState:
    """Resolved ORR kinetic quantities on geometric cell area."""

    catalyst_roughness_factor_cm2_pt_per_cm2_geo: float
    exchange_current_density_a_m2_geo: float
    activation_loss_v: float


def catalyst_roughness_factor(
    *,
    platinum_loading_mg_cm2_geo: float,
    ecsa_m2_pt_g_pt: float,
) -> float:
    """Return Pt roughness factor [cm2_Pt / cm2_geo].

    loading [mg/cm2_geo] * 1e-3 [g/mg]
    * ECSA [m2_Pt/g] * 1e4 [cm2_Pt/m2_Pt]
    = 10 * loading * ECSA.
    """
    if platinum_loading_mg_cm2_geo <= 0.0:
        raise ValueError("platinum_loading_mg_cm2_geo must be positive")
    if ecsa_m2_pt_g_pt <= 0.0:
        raise ValueError("ecsa_m2_pt_g_pt must be positive")
    return 10.0 * platinum_loading_mg_cm2_geo * ecsa_m2_pt_g_pt


def neyerlin_specific_exchange_current_density_a_cm2_pt(
    *,
    temperature_k: float,
    oxygen_partial_pressure_pa: float,
    reference: NeyerlinORRReference | None = None,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Return Pt-area-specific ORR exchange current density.

    i0(T,pO2) = i0* (pO2/p*)^gamma
                exp[-Ea/R * (1/T - 1/T*)].
    """
    ref = NeyerlinORRReference() if reference is None else reference
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if oxygen_partial_pressure_pa <= 0.0:
        raise ValueError("oxygen_partial_pressure_pa must be positive")
    if gas_constant_j_mol_k <= 0.0:
        raise ValueError("gas_constant_j_mol_k must be positive")

    pressure_factor = (
        oxygen_partial_pressure_pa / ref.reference_oxygen_pressure_pa
    ) ** ref.oxygen_reaction_order
    temperature_factor = np.exp(
        -ref.activation_energy_j_mol
        / gas_constant_j_mol_k
        * (
            1.0 / temperature_k
            - 1.0 / ref.reference_temperature_k
        )
    )
    return float(
        ref.reference_exchange_current_density_a_cm2_pt
        * pressure_factor
        * temperature_factor
    )


def geometric_exchange_current_density_a_m2(
    *,
    temperature_k: float,
    oxygen_partial_pressure_pa: float,
    platinum_loading_mg_cm2_geo: float,
    ecsa_m2_pt_g_pt: float,
    reference: NeyerlinORRReference | None = None,
) -> tuple[float, float]:
    """Return (roughness factor, geometric ORR exchange current density)."""
    roughness = catalyst_roughness_factor(
        platinum_loading_mg_cm2_geo=platinum_loading_mg_cm2_geo,
        ecsa_m2_pt_g_pt=ecsa_m2_pt_g_pt,
    )
    specific = neyerlin_specific_exchange_current_density_a_cm2_pt(
        temperature_k=temperature_k,
        oxygen_partial_pressure_pa=oxygen_partial_pressure_pa,
        reference=reference,
    )
    geometric_a_cm2 = specific * roughness
    return roughness, geometric_a_cm2 * 1.0e4


def cathodic_tafel_activation_loss_v(
    *,
    current_density_a_m2: float,
    exchange_current_density_a_m2: float,
    temperature_k: float,
    transfer_coefficient: float = 1.0,
    faraday_c_mol: float = 96485.33212,
    gas_constant_j_mol_k: float = 8.31446261815324,
) -> float:
    """Return cathodic ORR activation loss in the Tafel regime."""
    if current_density_a_m2 < 0.0:
        raise ValueError("current_density_a_m2 must be non-negative")
    if exchange_current_density_a_m2 <= 0.0:
        raise ValueError("exchange_current_density_a_m2 must be positive")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if transfer_coefficient <= 0.0:
        raise ValueError("transfer_coefficient must be positive")
    if faraday_c_mol <= 0.0 or gas_constant_j_mol_k <= 0.0:
        raise ValueError("physical constants must be positive")
    if current_density_a_m2 == 0.0:
        return 0.0
    if current_density_a_m2 <= exchange_current_density_a_m2:
        raise ValueError(
            "Tafel approximation requires current density above exchange current"
        )

    return float(
        gas_constant_j_mol_k
        * temperature_k
        / (transfer_coefficient * faraday_c_mol)
        * np.log(current_density_a_m2 / exchange_current_density_a_m2)
    )


def neyerlin_orr_activation_state(
    *,
    current_density_a_m2: float,
    temperature_k: float,
    oxygen_partial_pressure_pa: float,
    platinum_loading_mg_cm2_geo: float,
    ecsa_m2_pt_g_pt: float,
    reference: NeyerlinORRReference | None = None,
) -> ORRKineticState:
    """Return geometric j0 and Tafel activation loss from Neyerlin kinetics."""
    ref = NeyerlinORRReference() if reference is None else reference
    roughness, exchange = geometric_exchange_current_density_a_m2(
        temperature_k=temperature_k,
        oxygen_partial_pressure_pa=oxygen_partial_pressure_pa,
        platinum_loading_mg_cm2_geo=platinum_loading_mg_cm2_geo,
        ecsa_m2_pt_g_pt=ecsa_m2_pt_g_pt,
        reference=ref,
    )
    activation = cathodic_tafel_activation_loss_v(
        current_density_a_m2=current_density_a_m2,
        exchange_current_density_a_m2=exchange,
        temperature_k=temperature_k,
        transfer_coefficient=ref.cathodic_transfer_coefficient,
    )
    return ORRKineticState(
        catalyst_roughness_factor_cm2_pt_per_cm2_geo=roughness,
        exchange_current_density_a_m2_geo=exchange,
        activation_loss_v=activation,
    )
