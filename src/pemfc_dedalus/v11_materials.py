"""Literature-sourced FCgen-1020ACS material and MEA parameters for V11.

These values are deliberately kept separate from the historical CathodeParameters
defaults used by V0--V10 models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .membrane import membrane_proton_conductivity


@dataclass(frozen=True)
class V11MEAReference:
    """Non-fitted material/geometric parameters used by the V11 model."""

    active_cell_area_m2: float = 0.0145
    membrane_thickness_m: float = 80e-6
    gdl_thickness_m: float = 200e-6
    catalyst_layer_thickness_m: float = 10e-6

    gdl_porosity: float = 0.76
    catalyst_layer_porosity: float = 0.40

    membrane_dry_density_kg_m3: float = 1980.0
    membrane_equivalent_weight_kg_mol: float = 1.10

    gdl_intrinsic_permeability_m2: float = 1.0e-12
    catalyst_layer_intrinsic_permeability_m2: float = 1.0e-13
    liquid_water_contact_angle_deg: float = 110.0

    def current_density_a_m2(self, current_a: float) -> float:
        """Convert imposed stack current to per-cell geometric current density."""
        if current_a < 0.0:
            raise ValueError("current_a must be non-negative")
        return current_a / self.active_cell_area_m2

    def membrane_proton_conductivity_s_m(
        self,
        *,
        water_content: float,
        temperature_k: float,
    ) -> float:
        """Return Springer proton conductivity for the V11 membrane state."""
        conductivity = membrane_proton_conductivity(
            water_content,
            temperature_k,
        ).item()
        if not np.isfinite(conductivity) or conductivity <= 0.0:
            raise ValueError(
                "membrane proton conductivity must be finite and positive"
            )
        return float(conductivity)

    def membrane_area_specific_resistance_ohm_m2(
        self,
        *,
        water_content: float,
        temperature_k: float,
    ) -> float:
        """Return membrane ASR = L/sigma [ohm m2]."""
        conductivity = self.membrane_proton_conductivity_s_m(
            water_content=water_content,
            temperature_k=temperature_k,
        )
        return self.membrane_thickness_m / conductivity

    def membrane_ohmic_loss_v(
        self,
        *,
        current_a: float,
        water_content: float,
        temperature_k: float,
    ) -> float:
        """Return membrane ohmic voltage loss at imposed stack current."""
        current_density = self.current_density_a_m2(current_a)
        return (
            current_density
            * self.membrane_area_specific_resistance_ohm_m2(
                water_content=water_content,
                temperature_k=temperature_k,
            )
        )
