import pytest

from pemfc_dedalus.scaling import infer_active_area_scaling


def test_active_area_scaling_matches_reference_cell_current():
    scaling = infer_active_area_scaling(
        patch_area_m2=2.0e-6,
        patch_reference_current_a=2.0e-3,
        cell_reference_current_a=20.0,
    )

    assert scaling.area_scale_factor == pytest.approx(10000.0)
    assert scaling.inferred_active_area_m2 == pytest.approx(0.02)
    assert scaling.inferred_active_area_cm2 == pytest.approx(200.0)
    assert scaling.scale_patch_current(2.0e-3) == pytest.approx(20.0)


def test_active_area_scaling_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        infer_active_area_scaling(
            patch_area_m2=0.0,
            patch_reference_current_a=2.0e-3,
            cell_reference_current_a=20.0,
        )

    with pytest.raises(ValueError):
        infer_active_area_scaling(
            patch_area_m2=2.0e-6,
            patch_reference_current_a=0.0,
            cell_reference_current_a=20.0,
        )

    with pytest.raises(ValueError):
        infer_active_area_scaling(
            patch_area_m2=2.0e-6,
            patch_reference_current_a=2.0e-3,
            cell_reference_current_a=0.0,
        )
