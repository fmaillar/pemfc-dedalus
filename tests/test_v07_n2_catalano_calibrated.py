import pytest

from pemfc_dedalus.gas_permeability import barrer_to_si_permeability
from scripts.run_v07_n2_catalano_calibrated import (
    ANCHOR_PERMEABILITY_FACTOR,
    ANCHOR_WATER_ACTIVITY,
    make_calibrated_permeance_model,
    scalar_water_content,
)


def test_catalano_calibration_hits_humidity_anchor():
    dry_reference_si = barrer_to_si_permeability(0.18)
    model, anchors = make_calibrated_permeance_model(
        dry_reference_si=dry_reference_si,
        stack_temperature_k=308.15,
        gas_constant_j_mol_k=8.31446261815324,
        activation_energy_j_mol=0.0,
        cathode_relative_humidity=ANCHOR_WATER_ACTIVITY,
        membrane_thickness_m=50.0e-6,
        membrane_equivalent_weight_kg_mol=1.10,
        membrane_dry_density_kg_m3=2000.0,
        water_partial_molar_volume_m3_mol=17.0e-6,
    )

    dry_permeance = dry_reference_si / 50.0e-6
    anchor_permeance = model(ANCHOR_WATER_ACTIVITY)

    assert anchors["lambda_anchor"] == scalar_water_content(
        ANCHOR_WATER_ACTIVITY
    )
    assert anchor_permeance == pytest.approx(
        dry_permeance * ANCHOR_PERMEABILITY_FACTOR
    )
