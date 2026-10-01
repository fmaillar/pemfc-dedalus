"""Dedalus 1D streamwise verification model for V0.9."""

from __future__ import annotations

from dataclasses import dataclass

import dedalus.public as d3
import numpy as np

from .cathode_airflow import (
    molar_flow_from_slpm,
    oxygen_consumption_mol_s,
)


@dataclass(frozen=True)
class DedalusStreamwiseProfile:
    """Solved streamwise O2-flow and air-temperature fields."""

    streamwise_fraction: np.ndarray
    oxygen_molar_flow_mol_s: np.ndarray
    air_temperature_k: np.ndarray
    oxygen_mole_fraction: np.ndarray


def solve_streamwise_coupled_profile(
    *,
    current_a: float,
    total_air_flow_slpm: float,
    n_cells: int,
    oxygen_mole_fraction: float,
    faraday_c_mol: float,
    inlet_temperature_k: float,
    total_heat_rejection_w: float,
    points: int = 64,
    air_density_kg_m3: float = 1.204,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> DedalusStreamwiseProfile:
    """Solve reduced coupled O2/thermal balances on xi in [0, 1]."""
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if points < 4:
        raise ValueError("points must be >= 4")
    if inlet_temperature_k <= 0.0:
        raise ValueError("inlet_temperature_k must be positive")
    if total_heat_rejection_w < 0.0:
        raise ValueError("total_heat_rejection_w must be non-negative")
    if not 0.0 < oxygen_mole_fraction < 1.0:
        raise ValueError("oxygen_mole_fraction must lie in (0, 1)")
    if air_density_kg_m3 <= 0.0:
        raise ValueError("air_density_kg_m3 must be positive")
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    per_cell_air_flow_slpm = total_air_flow_slpm / n_cells
    inlet_total_molar_flow = molar_flow_from_slpm(per_cell_air_flow_slpm)
    inlet_oxygen_flow = oxygen_mole_fraction * inlet_total_molar_flow
    inlet_inert_flow = (1.0 - oxygen_mole_fraction) * inlet_total_molar_flow
    oxygen_consumption = oxygen_consumption_mol_s(
        current_a,
        faraday_c_mol=faraday_c_mol,
    )
    if oxygen_consumption > inlet_oxygen_flow:
        raise ValueError("airflow cannot supply the requested oxygen consumption")

    per_cell_volumetric_flow_m3_s = per_cell_air_flow_slpm * 1.0e-3 / 60.0
    per_cell_mass_flow_kg_s = (
        per_cell_volumetric_flow_m3_s * air_density_kg_m3
    )
    heat_rejection_w_per_cell = total_heat_rejection_w / n_cells
    if per_cell_mass_flow_kg_s <= 0.0:
        if heat_rejection_w_per_cell > 0.0:
            raise ValueError("airflow must be positive when heat is rejected")
        temperature_rise_k = 0.0
    else:
        temperature_rise_k = heat_rejection_w_per_cell / (
            per_cell_mass_flow_kg_s * air_specific_heat_j_kg_k
        )

    coords = d3.CartesianCoordinates("xi")
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.ChebyshevT(
        coords["xi"],
        size=points,
        bounds=(0.0, 1.0),
        dealias=3 / 2,
    )

    oxygen_flow = dist.Field(name="oxygen_flow", bases=basis)
    temperature = dist.Field(name="temperature", bases=basis)
    tau_oxygen = dist.Field(name="tau_oxygen")
    tau_temperature = dist.Field(name="tau_temperature")

    lift_basis = basis.derivative_basis(1)

    def lift(field):
        return d3.Lift(field, lift_basis, -1)

    def dxi(field):
        return d3.Differentiate(field, coords["xi"])

    problem = d3.LBVP(
        [oxygen_flow, temperature, tau_oxygen, tau_temperature],
        namespace=locals(),
    )
    problem.add_equation(
        "dxi(oxygen_flow) + lift(tau_oxygen) = -oxygen_consumption"
    )
    problem.add_equation("oxygen_flow(xi=0) = inlet_oxygen_flow")
    problem.add_equation(
        "dxi(temperature) + lift(tau_temperature) = temperature_rise_k"
    )
    problem.add_equation("temperature(xi=0) = inlet_temperature_k")

    solver = problem.build_solver()
    solver.solve()

    xi = np.asarray(dist.local_grid(basis), dtype=float)
    oxygen_values = np.asarray(oxygen_flow["g"], dtype=float).copy()
    temperature_values = np.asarray(temperature["g"], dtype=float).copy()
    total_molar_flow = oxygen_values + inlet_inert_flow
    local_oxygen_fraction = oxygen_values / total_molar_flow

    order = np.argsort(xi)
    return DedalusStreamwiseProfile(
        streamwise_fraction=xi[order],
        oxygen_molar_flow_mol_s=oxygen_values[order],
        air_temperature_k=temperature_values[order],
        oxygen_mole_fraction=local_oxygen_fraction[order],
    )



def solve_streamwise_finite_thermal_profile(
    *,
    current_a: float,
    total_air_flow_slpm: float,
    n_cells: int,
    oxygen_mole_fraction: float,
    faraday_c_mol: float,
    inlet_temperature_k: float,
    stack_temperature_k: float,
    ntu: float,
    points: int = 64,
) -> DedalusStreamwiseProfile:
    """Solve O2 balance and finite-NTU streamwise air heating."""
    if ntu < 0.0:
        raise ValueError("ntu must be non-negative")
    if stack_temperature_k <= 0.0:
        raise ValueError("stack_temperature_k must be positive")

    per_cell_air_flow_slpm = total_air_flow_slpm / n_cells
    inlet_total_molar_flow = molar_flow_from_slpm(per_cell_air_flow_slpm)
    inlet_oxygen_flow = oxygen_mole_fraction * inlet_total_molar_flow
    inlet_inert_flow = (1.0 - oxygen_mole_fraction) * inlet_total_molar_flow
    oxygen_consumption = oxygen_consumption_mol_s(
        current_a,
        faraday_c_mol=faraday_c_mol,
    )
    if oxygen_consumption > inlet_oxygen_flow:
        raise ValueError("airflow cannot supply the requested oxygen consumption")

    coords = d3.CartesianCoordinates("xi")
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.ChebyshevT(
        coords["xi"],
        size=points,
        bounds=(0.0, 1.0),
        dealias=3 / 2,
    )

    oxygen_flow = dist.Field(name="oxygen_flow", bases=basis)
    temperature = dist.Field(name="temperature", bases=basis)
    tau_oxygen = dist.Field(name="tau_oxygen")
    tau_temperature = dist.Field(name="tau_temperature")

    lift_basis = basis.derivative_basis(1)

    def lift(field):
        return d3.Lift(field, lift_basis, -1)

    def dxi(field):
        return d3.Differentiate(field, coords["xi"])

    problem = d3.LBVP(
        [oxygen_flow, temperature, tau_oxygen, tau_temperature],
        namespace=locals(),
    )
    problem.add_equation(
        "dxi(oxygen_flow) + lift(tau_oxygen) = -oxygen_consumption"
    )
    problem.add_equation("oxygen_flow(xi=0) = inlet_oxygen_flow")
    problem.add_equation(
        "dxi(temperature) + ntu*temperature + lift(tau_temperature) "
        "= ntu*stack_temperature_k"
    )
    problem.add_equation("temperature(xi=0) = inlet_temperature_k")

    solver = problem.build_solver()
    solver.solve()

    xi = np.asarray(dist.local_grid(basis), dtype=float)
    oxygen_values = np.asarray(oxygen_flow["g"], dtype=float).copy()
    temperature_values = np.asarray(temperature["g"], dtype=float).copy()
    total_molar_flow = oxygen_values + inlet_inert_flow
    local_oxygen_fraction = oxygen_values / total_molar_flow

    order = np.argsort(xi)
    return DedalusStreamwiseProfile(
        streamwise_fraction=xi[order],
        oxygen_molar_flow_mol_s=oxygen_values[order],
        air_temperature_k=temperature_values[order],
        oxygen_mole_fraction=local_oxygen_fraction[order],
    )
