"""State-space interface for the V11 galvanostatic PEMFC digital twin.

The dynamic state itself remains V11DynamicState. This module only classifies
model quantities into manipulated inputs, exogenous disturbances, model
parameters and model outputs. It contains no control law and makes no claim\nabout sensor availability or formal observability.
"""

from __future__ import annotations

from dataclasses import dataclass

from .v11_runner import V11RunnerInputs, V11TrajectoryPoint\nfrom .v11_system import V11DynamicState


@dataclass(frozen=True)
class V11ManipulatedInputs:
    """Inputs that an external controller may manipulate."""

    stack_air_flow_slpm: float
    purge_event: bool


@dataclass(frozen=True)
class V11ExogenousInputs:
    """Load and boundary-condition disturbances imposed on the model."""

    current_a: float
    inlet_air_temperature_k: float
    cathode_total_pressure_pa: float
    inlet_oxygen_mole_fraction: float
    inlet_water_mole_fraction: float


@dataclass(frozen=True)
class V11ModelParameters:
    """Electrochemical parameters, not control inputs."""

    cathode_platinum_loading_mg_cm2_geo: float
    cathode_ecsa_m2_pt_g_pt: float
    membrane_conductivity_multiplier: float = 1.0
    additional_resolved_loss_v: float = 0.0


@dataclass(frozen=True)
class V11ModelOutputs:
    """Algebraic/model outputs y; sensor availability is defined separately."""

    stack_temperature_k: float
    cell_voltage_v: float
    air_outlet_temperature_k: float
    anode_water_activity: float
    cathode_water_activity: float


def model_outputs_from_trajectory_point(
    point: V11TrajectoryPoint,
) -> V11ModelOutputs:
    """Build the model-output vector from one predictive trajectory sample."""
    diagnostics = point.diagnostics
    if diagnostics.voltage is None:
        raise ValueError("predictive trajectory requires voltage diagnostics")
    if diagnostics.heat_transfer is None:
        raise ValueError("predictive trajectory requires heat-transfer diagnostics")

    return V11ModelOutputs(
        stack_temperature_k=point.state.stack_temperature_k,
        cell_voltage_v=diagnostics.voltage.cell_voltage_v,
        air_outlet_temperature_k=diagnostics.heat_transfer.outlet_temperature_k,
        anode_water_activity=diagnostics.anode_water_activity,
        cathode_water_activity=diagnostics.cathode_water_activity,
    )


@dataclass(frozen=True)
class V11StateSpaceSample:
    """One time sample of the nonlinear hybrid state-space model."""

    time_s: float
    state: V11DynamicState
    manipulated: V11ManipulatedInputs
    exogenous: V11ExogenousInputs
    parameters: V11ModelParameters
    outputs: V11ModelOutputs


def state_space_sample_from_trajectory_point(
    point: V11TrajectoryPoint,
    inputs: V11RunnerInputs,
) -> V11StateSpaceSample:
    """Expose one runner sample as (t, x, u, d, p, y)."""
    manipulated = V11ManipulatedInputs(
        stack_air_flow_slpm=point.stack_air_flow_slpm,
        purge_event=point.purge_event,
    )
    exogenous = V11ExogenousInputs(
        current_a=point.current_a,
        inlet_air_temperature_k=inputs.inlet_air_temperature_k,
        cathode_total_pressure_pa=inputs.cathode_total_pressure_pa,
        inlet_oxygen_mole_fraction=inputs.inlet_oxygen_mole_fraction,
        inlet_water_mole_fraction=inputs.inlet_water_mole_fraction,
    )
    parameters = V11ModelParameters(
        cathode_platinum_loading_mg_cm2_geo=(
            inputs.cathode_platinum_loading_mg_cm2_geo
        ),
        cathode_ecsa_m2_pt_g_pt=inputs.cathode_ecsa_m2_pt_g_pt,
        membrane_conductivity_multiplier=(
            inputs.membrane_conductivity_multiplier
        ),
        additional_resolved_loss_v=inputs.additional_resolved_loss_v,
    )
    return V11StateSpaceSample(
        time_s=point.time_s,
        state=point.state,
        manipulated=manipulated,
        exogenous=exogenous,
        parameters=parameters,
        outputs=model_outputs_from_trajectory_point(point),
    )
