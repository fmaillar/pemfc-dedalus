import pytest

from pemfc_dedalus.gas_permeability import barrer_to_si_permeability
from scripts.run_v07_n2_catalano_membrane_transport import (
    make_v06_transport_permeance_model,
)


def test_v06_transport_permeance_model_is_positive_and_current_dependent():
    model = make_v06_transport_permeance_model(
        dry_reference_si=barrer_to_si_permeability(0.18),
        stack_temperature_k=312.9612,
        gas_constant_j_mol_k=8.31446261815324,
        faraday_c_mol=96485.33212,
        fixed_charge_mol_m3=2000.0 / 1.10,
        activation_energy_j_mol=49_600.0,
        cathode_relative_humidity=0.50,
        membrane_thickness_m=50.0e-6,
        membrane_equivalent_weight_kg_mol=1.10,
        membrane_dry_density_kg_m3=2000.0,
        membrane_water_diffusivity_m2_s=2.0e-10,
        anode_transfer_coefficient_m_s=2.0e-6,
        water_partial_molar_volume_m3_mol=17.0e-6,
        profile_points=65,
    )

    low_current = model(0.50, 100.0)
    high_current = model(0.50, 1000.0)

    assert low_current > 0.0
    assert high_current > 0.0
    assert high_current != pytest.approx(low_current)


def test_v06_transport_permeance_model_rejects_single_point_profile():
    with pytest.raises(ValueError, match="profile_points"):
        make_v06_transport_permeance_model(
            dry_reference_si=barrer_to_si_permeability(0.18),
            stack_temperature_k=312.9612,
            gas_constant_j_mol_k=8.31446261815324,
            faraday_c_mol=96485.33212,
            fixed_charge_mol_m3=2000.0 / 1.10,
            activation_energy_j_mol=49_600.0,
            cathode_relative_humidity=0.50,
            membrane_thickness_m=50.0e-6,
            membrane_equivalent_weight_kg_mol=1.10,
            membrane_dry_density_kg_m3=2000.0,
            membrane_water_diffusivity_m2_s=2.0e-10,
            anode_transfer_coefficient_m_s=2.0e-6,
            water_partial_molar_volume_m3_mol=17.0e-6,
            profile_points=1,
        )
