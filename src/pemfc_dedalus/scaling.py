"""Active-area scaling helpers for the V0.7 full-cell anode model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveAreaScaling:
    """Map a representative numerical patch onto one full PEMFC cell."""

    patch_area_m2: float
    patch_reference_current_a: float
    cell_reference_current_a: float
    area_scale_factor: float
    inferred_active_area_m2: float

    @property
    def inferred_active_area_cm2(self) -> float:
        return self.inferred_active_area_m2 * 1.0e4

    def scale_patch_current(self, patch_current_a: float) -> float:
        """Scale a patch-integrated current to the inferred full-cell area."""
        return patch_current_a * self.area_scale_factor


def infer_active_area_scaling(
    *,
    patch_area_m2: float,
    patch_reference_current_a: float,
    cell_reference_current_a: float,
) -> ActiveAreaScaling:
    """Infer full-cell active area from a reference patch current density.

    The model assumes that the representative patch current density is uniform
    over the inferred active area:

        j_ref = I_patch,ref / A_patch
        A_active = I_cell,ref / j_ref

    This is a model-derived scaling calibration, not a measured Ballard active
    area.
    """
    if patch_area_m2 <= 0.0:
        raise ValueError("patch_area_m2 must be positive")
    if patch_reference_current_a <= 0.0:
        raise ValueError("patch_reference_current_a must be positive")
    if cell_reference_current_a <= 0.0:
        raise ValueError("cell_reference_current_a must be positive")

    area_scale_factor = cell_reference_current_a / patch_reference_current_a
    inferred_active_area_m2 = patch_area_m2 * area_scale_factor
    return ActiveAreaScaling(
        patch_area_m2=patch_area_m2,
        patch_reference_current_a=patch_reference_current_a,
        cell_reference_current_a=cell_reference_current_a,
        area_scale_factor=area_scale_factor,
        inferred_active_area_m2=inferred_active_area_m2,
    )
