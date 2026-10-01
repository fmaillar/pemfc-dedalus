"""Physics-first lumped thermal balance for V11.

The stack energy balance is written without an empirical thermal time constant.
Heat generation follows the Ballard 1020ACS manufacturer relation and sensible
air cooling is evaluated from the inlet/outlet enthalpy rise.

The air outlet temperature is an explicit physical quantity. A separate helper
provides the ideal upper-bound closure T_out = T_stack; this is not treated as
the final heat-transfer model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalBalance:
    """Instantaneous stack thermal powers and temperature derivative."""

    heat_generation_w: float
    air_cooling_w: float
    net_heat_w: float
    temperature_rate_k_s: float


def stack_heat_generation_w(
    *,
    n_cells: int,
    current_a: float,
    cell_voltage_v: float,
    manufacturer_heat_voltage_v: float = 1.253,
) -> float:
    """Return Ballard 1020ACS stack heat generation [W].

    Manufacturer manual relation:

        Q_gen = n_cells * I * (1.253 - V_cell).
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if cell_voltage_v < 0.0:
        raise ValueError("cell_voltage_v must be non-negative")
    if manufacturer_heat_voltage_v <= 0.0:
        raise ValueError("manufacturer_heat_voltage_v must be positive")

    heat = n_cells * current_a * (
        manufacturer_heat_voltage_v - cell_voltage_v
    )
    if heat < 0.0:
        raise ValueError(
            "manufacturer heat relation gives negative heat generation"
        )
    return heat


def standard_air_mass_flow_kg_s(
    *,
    stack_air_flow_slpm: float,
    standard_air_density_kg_m3: float = 1.293,
) -> float:
    """Convert stack standard-air flow to mass flow [kg/s].

    The default density is the Ballard manual value for air at 0 degC, 1 atm.
    """
    if stack_air_flow_slpm < 0.0:
        raise ValueError("stack_air_flow_slpm must be non-negative")
    if standard_air_density_kg_m3 <= 0.0:
        raise ValueError("standard_air_density_kg_m3 must be positive")

    standard_volume_flow_m3_s = stack_air_flow_slpm * 1.0e-3 / 60.0
    return standard_air_density_kg_m3 * standard_volume_flow_m3_s


def sensible_air_cooling_w(
    *,
    stack_air_flow_slpm: float,
    inlet_temperature_k: float,
    outlet_temperature_k: float,
    air_specific_heat_j_kg_k: float = 1005.0,
    standard_air_density_kg_m3: float = 1.293,
) -> float:
    """Return sensible heat carried away by the cathode air stream [W]."""
    if inlet_temperature_k <= 0.0 or outlet_temperature_k <= 0.0:
        raise ValueError("air temperatures must be positive")
    if outlet_temperature_k < inlet_temperature_k:
        raise ValueError(
            "outlet_temperature_k must not be below inlet_temperature_k"
        )
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    mass_flow = standard_air_mass_flow_kg_s(
        stack_air_flow_slpm=stack_air_flow_slpm,
        standard_air_density_kg_m3=standard_air_density_kg_m3,
    )
    return (
        mass_flow
        * air_specific_heat_j_kg_k
        * (outlet_temperature_k - inlet_temperature_k)
    )


def stack_temperature_rhs_k_s(
    *,
    n_cells: int,
    current_a: float,
    cell_voltage_v: float,
    stack_air_flow_slpm: float,
    inlet_temperature_k: float,
    outlet_temperature_k: float,
    thermal_mass_j_k_per_cell: float = 100.0,
    air_specific_heat_j_kg_k: float = 1005.0,
    standard_air_density_kg_m3: float = 1.293,
) -> ThermalBalance:
    """Return the V11 lumped stack temperature derivative.

    Thermal storage is

        C_stack dT/dt = Q_gen - Q_air,

    with C_stack = n_cells * C_cell.
    """
    if thermal_mass_j_k_per_cell <= 0.0:
        raise ValueError("thermal_mass_j_k_per_cell must be positive")

    heat_generation = stack_heat_generation_w(
        n_cells=n_cells,
        current_a=current_a,
        cell_voltage_v=cell_voltage_v,
    )
    air_cooling = sensible_air_cooling_w(
        stack_air_flow_slpm=stack_air_flow_slpm,
        inlet_temperature_k=inlet_temperature_k,
        outlet_temperature_k=outlet_temperature_k,
        air_specific_heat_j_kg_k=air_specific_heat_j_kg_k,
        standard_air_density_kg_m3=standard_air_density_kg_m3,
    )
    net_heat = heat_generation - air_cooling
    thermal_mass = n_cells * thermal_mass_j_k_per_cell

    return ThermalBalance(
        heat_generation_w=heat_generation,
        air_cooling_w=air_cooling,
        net_heat_w=net_heat,
        temperature_rate_k_s=net_heat / thermal_mass,
    )


def ideal_equilibrated_air_temperature_rhs_k_s(
    *,
    n_cells: int,
    current_a: float,
    cell_voltage_v: float,
    stack_temperature_k: float,
    stack_air_flow_slpm: float,
    inlet_temperature_k: float,
    thermal_mass_j_k_per_cell: float = 100.0,
    air_specific_heat_j_kg_k: float = 1005.0,
    standard_air_density_kg_m3: float = 1.293,
) -> ThermalBalance:
    """Return ideal upper-bound cooling with T_air,out = T_stack.

    This helper is a diagnostic limiting case, not the final V11 heat-transfer
    closure.
    """
    if stack_temperature_k < inlet_temperature_k:
        raise ValueError(
            "ideal equilibrated-air limit requires stack temperature "
            "not below inlet temperature"
        )

    return stack_temperature_rhs_k_s(
        n_cells=n_cells,
        current_a=current_a,
        cell_voltage_v=cell_voltage_v,
        stack_air_flow_slpm=stack_air_flow_slpm,
        inlet_temperature_k=inlet_temperature_k,
        outlet_temperature_k=stack_temperature_k,
        thermal_mass_j_k_per_cell=thermal_mass_j_k_per_cell,
        air_specific_heat_j_kg_k=air_specific_heat_j_kg_k,
        standard_air_density_kg_m3=standard_air_density_kg_m3,
    )
