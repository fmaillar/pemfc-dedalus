"""Tests for thermo-coupled local O2 state."""

from __future__ import annotations

import numpy as np
import pytest

from pemfc_dedalus.cathode_airflow import ideal_gas_species_concentration_mol_m3


def test_ideal_gas_species_concentration_scalar() -> None:
    value = ideal_gas_species_concentration_mol_m3(
        mole_fraction=0.21,
        pressure_pa=101325.0,
        temperature_k=300.0,
        gas_constant_j_mol_k=8.31446261815324,
    )
    assert value == pytest.approx(0.21 * 101325.0 / (8.31446261815324 * 300.0))


def test_species_concentration_decreases_with_temperature() -> None:
    values = ideal_gas_species_concentration_mol_m3(
        mole_fraction=0.21,
        pressure_pa=101325.0,
        temperature_k=np.array([290.0, 300.0, 310.0]),
        gas_constant_j_mol_k=8.31446261815324,
    )
    assert np.all(np.diff(values) < 0.0)


def test_species_concentration_rejects_invalid_mole_fraction() -> None:
    with pytest.raises(ValueError, match="mole_fraction"):
        ideal_gas_species_concentration_mol_m3(
            mole_fraction=1.1,
            pressure_pa=101325.0,
            temperature_k=300.0,
            gas_constant_j_mol_k=8.31446261815324,
        )
