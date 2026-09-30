import numpy as np
import pytest

from scripts.analyze_v07_n2_membrane_profiles import resistance_fraction


def test_resistance_fraction_splits_uniform_profile_in_half():
    z = np.linspace(0.0, 1.0, 65)
    permeability = np.ones_like(z)

    anode = resistance_fraction(
        z,
        permeability,
        start_fraction=0.0,
        end_fraction=0.5,
    )
    cathode = resistance_fraction(
        z,
        permeability,
        start_fraction=0.5,
        end_fraction=1.0,
    )

    assert anode == pytest.approx(0.5)
    assert cathode == pytest.approx(0.5)


def test_resistance_fraction_rejects_invalid_interval():
    z = np.linspace(0.0, 1.0, 65)
    permeability = np.ones_like(z)

    with pytest.raises(ValueError):
        resistance_fraction(
            z,
            permeability,
            start_fraction=0.7,
            end_fraction=0.3,
        )
