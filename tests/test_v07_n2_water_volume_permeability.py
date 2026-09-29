from pemfc_dedalus.gas_permeability import (
    barrer_to_si_permeability,
)
from scripts.run_v07_n2_water_volume_permeability import (
    make_water_volume_permeance_model,
    scalar_water_content,
)


def test_water_volume_permeance_model_increases_with_anode_rh():
    dry_reference_si = barrer_to_si_permeability(0.24)
    model, lambda_cathode = make_water_volume_permeance_model(
        dry_reference_si=dry_reference_si,
        stack_temperature_k=312.9612,
        gas_constant_j_mol_k=8.31446261815324,
        cathode_relative_humidity=0.5,
        membrane_thickness_m=50.0e-6,
        membrane_equivalent_weight_kg_mol=1.10,
        membrane_dry_density_kg_m3=2000.0,
    )

    dry_side = model(0.0)
    wet_side = model(0.9)

    assert lambda_cathode == scalar_water_content(0.5)
    assert wet_side > dry_side
