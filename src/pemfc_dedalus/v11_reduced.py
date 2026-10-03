"""Minimal two-state V11 model for estimation-oriented studies.

The full eight-state V11 plant remains the physical reference.  This reduced
model keeps only the two slow states that directly matter for the available
measurements:

    x_r = [stack temperature, mean membrane water content].

Current, cathode-air flow and anode hydrogen pressure are measured inputs.
Cathode gas composition is closed algebraically from inlet flow and Faraday
stoichiometry.  No gas-water inventory is estimated.  The membrane-water
balance therefore uses the cathode interface only in this first reduced model;
that approximation is explicit and can be relaxed later if validation requires
it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anode import water_saturation_pressure_pa
from .ballard_1020acs import UserStackConfiguration
from .v11_cathode import standard_litre_per_minute_to_mol_s
from .v11_heat_transfer import cathode_air_outlet_temperature_ballard_v11
from .v11_membrane import (
    ge_interface_water_flux_into_membrane,
    membrane_hydration_rhs,
)
from .v11_thermal import stack_temperature_rhs_k_s
from .v11_voltage import predict_cell_voltage_v

FARADAY_C_MOL = 96485.33212
REDUCED_DEFAULT_DT_S = 0.1


@dataclass(frozen=True)
class V11ReducedState:
    """Minimal dynamic state used by the first reduced model."""

    stack_temperature_k: float
    membrane_mean_water_content: float


@dataclass(frozen=True)
class V11ReducedInputs:
    """Measured or known inputs to the reduced model."""

    current_a: float
    stack_air_flow_slpm: float
    anode_hydrogen_pressure_pa: float
    inlet_air_temperature_k: float
    cathode_total_pressure_pa: float
    inlet_oxygen_mole_fraction: float
    inlet_water_mole_fraction: float


@dataclass(frozen=True)
class V11ReducedParameters:
    """Electrochemical parameters retained from the full V11 model."""

    cathode_platinum_loading_mg_cm2_geo: float
    cathode_ecsa_m2_pt_g_pt: float
    membrane_conductivity_multiplier: float = 1.0
    additional_resolved_loss_v: float = 0.0


@dataclass(frozen=True)
class V11ReducedDerivative:
    """Time derivative of the minimal two-state model."""

    stack_temperature_k_s: float
    membrane_mean_water_content_s: float


@dataclass(frozen=True)
class V11ReducedOutputs:
    """Measured-model outputs and algebraic diagnostics."""

    stack_voltage_v: float
    stack_temperature_k: float
    cathode_oxygen_partial_pressure_pa: float
    cathode_water_activity: float
    cathode_outlet_water_mole_fraction: float


def _bounded_activity(
    *,
    water_partial_pressure_pa: float,
    temperature_k: float,
) -> float:
    saturation = water_saturation_pressure_pa(temperature_k)
    return min(max(water_partial_pressure_pa / saturation, 1.0e-9), 1.0)


def _algebraic_cathode_state(
    *,
    current_a: float,
    stack_air_flow_slpm: float,
    n_cells: int,
    cathode_total_pressure_pa: float,
    inlet_oxygen_mole_fraction: float,
    inlet_water_mole_fraction: float,
    temperature_k: float,
) -> tuple[float, float, float]:
    """Return outlet pO2, water activity and outlet water fraction.

    The closure assumes quasi-steady cathode gas composition.  Oxygen
    consumption and water production follow Faraday stoichiometry exactly.
    Membrane water storage is not included in this algebraic gas balance.
    """
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if stack_air_flow_slpm <= 0.0:
        raise ValueError("stack_air_flow_slpm must be positive")
    if cathode_total_pressure_pa <= 0.0:
        raise ValueError("cathode_total_pressure_pa must be positive")
    if not 0.0 < inlet_oxygen_mole_fraction < 1.0:
        raise ValueError("inlet_oxygen_mole_fraction must lie in (0, 1)")
    if not 0.0 <= inlet_water_mole_fraction < 1.0:
        raise ValueError("inlet_water_mole_fraction must lie in [0, 1)")
    if inlet_oxygen_mole_fraction + inlet_water_mole_fraction >= 1.0:
        raise ValueError("inlet wet-gas mole fractions must sum below one")

    inlet_per_cell = standard_litre_per_minute_to_mol_s(
        stack_air_flow_slpm / n_cells
    )
    oxygen_in = inlet_per_cell * inlet_oxygen_mole_fraction
    water_in = inlet_per_cell * inlet_water_mole_fraction
    oxygen_consumed = current_a / (4.0 * FARADAY_C_MOL)
    water_produced = current_a / (2.0 * FARADAY_C_MOL)

    oxygen_out = oxygen_in - oxygen_consumed
    if oxygen_out <= 0.0:
        raise ValueError("reduced cathode closure predicts oxygen starvation")

    nitrogen_out = inlet_per_cell * (
        1.0 - inlet_oxygen_mole_fraction - inlet_water_mole_fraction
    )
    water_out = water_in + water_produced
    total_out = oxygen_out + nitrogen_out + water_out

    oxygen_fraction = oxygen_out / total_out
    water_fraction = water_out / total_out
    oxygen_partial_pressure = oxygen_fraction * cathode_total_pressure_pa
    water_activity = _bounded_activity(
        water_partial_pressure_pa=water_fraction * cathode_total_pressure_pa,
        temperature_k=temperature_k,
    )
    return oxygen_partial_pressure, water_activity, water_fraction


def reduced_v11_rhs(
    *,
    state: V11ReducedState,
    inputs: V11ReducedInputs,
    parameters: V11ReducedParameters,
    stack: UserStackConfiguration | None = None,
) -> tuple[V11ReducedDerivative, V11ReducedOutputs]:
    """Evaluate the minimal two-state reduced V11 model."""
    if state.stack_temperature_k <= 0.0:
        raise ValueError("stack_temperature_k must be positive")
    if state.membrane_mean_water_content < 0.0:
        raise ValueError("membrane_mean_water_content must be non-negative")
    if inputs.anode_hydrogen_pressure_pa <= 0.0:
        raise ValueError("anode_hydrogen_pressure_pa must be positive")

    cfg = UserStackConfiguration() if stack is None else stack
    oxygen_pressure, cathode_activity, water_fraction = (
        _algebraic_cathode_state(
            current_a=inputs.current_a,
            stack_air_flow_slpm=inputs.stack_air_flow_slpm,
            n_cells=cfg.n_cells,
            cathode_total_pressure_pa=inputs.cathode_total_pressure_pa,
            inlet_oxygen_mole_fraction=inputs.inlet_oxygen_mole_fraction,
            inlet_water_mole_fraction=inputs.inlet_water_mole_fraction,
            temperature_k=state.stack_temperature_k,
        )
    )

    voltage = predict_cell_voltage_v(
        current_a=inputs.current_a,
        temperature_k=state.stack_temperature_k,
        hydrogen_partial_pressure_pa=inputs.anode_hydrogen_pressure_pa,
        oxygen_partial_pressure_pa=oxygen_pressure,
        water_activity=cathode_activity,
        membrane_mean_water_content=state.membrane_mean_water_content,
        cathode_platinum_loading_mg_cm2_geo=(
            parameters.cathode_platinum_loading_mg_cm2_geo
        ),
        cathode_ecsa_m2_pt_g_pt=parameters.cathode_ecsa_m2_pt_g_pt,
        membrane_conductivity_multiplier=(
            parameters.membrane_conductivity_multiplier
        ),
        additional_resolved_loss_v=parameters.additional_resolved_loss_v,
    )

    heat_transfer = cathode_air_outlet_temperature_ballard_v11(
        stack_temperature_k=state.stack_temperature_k,
        inlet_temperature_k=inputs.inlet_air_temperature_k,
        stack_air_flow_slpm=inputs.stack_air_flow_slpm,
        n_cells=cfg.n_cells,
    )
    thermal = stack_temperature_rhs_k_s(
        n_cells=cfg.n_cells,
        current_a=inputs.current_a,
        cell_voltage_v=voltage.cell_voltage_v,
        stack_air_flow_slpm=inputs.stack_air_flow_slpm,
        inlet_temperature_k=inputs.inlet_air_temperature_k,
        outlet_temperature_k=heat_transfer.outlet_temperature_k,
    )

    cathode_interface = ge_interface_water_flux_into_membrane(
        gas_water_activity=cathode_activity,
        membrane_water_content=state.membrane_mean_water_content,
        temperature_k=state.stack_temperature_k,
    )
    membrane = membrane_hydration_rhs(
        anode_interface_flux_into_membrane_mol_m2_s=0.0,
        cathode_interface_flux_into_membrane_mol_m2_s=(
            cathode_interface.flux_into_membrane_mol_m2_s
        ),
    )

    derivative = V11ReducedDerivative(
        stack_temperature_k_s=thermal.temperature_rate_k_s,
        membrane_mean_water_content_s=membrane.mean_water_content_rate_s,
    )
    outputs = V11ReducedOutputs(
        stack_voltage_v=cfg.stack_voltage_v(voltage.cell_voltage_v),
        stack_temperature_k=state.stack_temperature_k,
        cathode_oxygen_partial_pressure_pa=oxygen_pressure,
        cathode_water_activity=cathode_activity,
        cathode_outlet_water_mole_fraction=water_fraction,
    )
    return derivative, outputs


def advance_reduced_v11_euler(
    *,
    state: V11ReducedState,
    inputs: V11ReducedInputs,
    parameters: V11ReducedParameters,
    dt_s: float = REDUCED_DEFAULT_DT_S,
    stack: UserStackConfiguration | None = None,
) -> V11ReducedState:
    """Advance the reduced state by one explicit-Euler step.

    The default 0.1 s step is selected from a local convergence study over
    5--26.04 A. After 30 s it keeps the stack-voltage deviation within about
    1.1 mV of a 1 ms reference while remaining inexpensive. Callers may
    override dt_s for convergence studies or externally sampled data.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    derivative, _ = reduced_v11_rhs(
        state=state,
        inputs=inputs,
        parameters=parameters,
        stack=stack,
    )
    next_state = V11ReducedState(
        stack_temperature_k=(
            state.stack_temperature_k
            + dt_s * derivative.stack_temperature_k_s
        ),
        membrane_mean_water_content=(
            state.membrane_mean_water_content
            + dt_s * derivative.membrane_mean_water_content_s
        ),
    )
    if next_state.stack_temperature_k <= 0.0:
        raise ValueError("Euler step produced non-positive stack temperature")
    if next_state.membrane_mean_water_content < 0.0:
        raise ValueError("Euler step produced negative membrane water content")
    return next_state
