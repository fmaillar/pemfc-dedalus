from pemfc_dedalus.gas_permeability import barrer_to_si_permeability
from scripts.run_v07_n2_water_content_permeability import (
    make_water_content_permeance_model,
    scalar_water_content,
)


def test_scalar_water_content_increases_with_activity():
    assert scalar_water_content(0.9) > scalar_water_content(0.5)
    assert scalar_water_content(0.5) > scalar_water_content(0.0)


def test_water_content_permeance_model_increases_with_anode_rh():
    dry_reference_si = barrer_to_si_permeability(0.24)
    model, lambda_dry, lambda_reference, lambda_cathode = (
        make_water_content_permeance_model(
            dry_reference_si=dry_reference_si,
            stack_temperature_k=312.9612,
            gas_constant_j_mol_k=8.31446261815324,
            cathode_relative_humidity=0.5,
            membrane_thickness_m=50.0e-6,
        )
    )

    dry_side = model(0.0)
    wet_side = model(0.9)

    assert lambda_reference > lambda_cathode > lambda_dry
    assert wet_side > dry_side
