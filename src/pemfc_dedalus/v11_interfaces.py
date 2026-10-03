"""State-space interface for the V11 galvanostatic PEMFC digital twin.

The dynamic state itself remains V11DynamicState. This module classifies
quantities into manipulated inputs, exogenous disturbances, model parameters,
model outputs and the actual sensor measurements available on the stack.
It contains no control law and makes no claim about formal observability.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ballard_1020acs import UserStackConfiguration
from .v11_runner import V11RunnerInputs, V11TrajectoryPoint
from .v11_system import V11DynamicState


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
    """Algebraic/model outputs y_model."""

    stack_temperature_k: float
    cell_voltage_v: float
    stack_voltage_v: float
    air_outlet_temperature_k: float
    anode_water_activity: float
    cathode_water_activity: float


@dataclass(frozen=True)
class V11Measurements:
    """Actual measured quantities available on the physical stack."""

    current_a: float
    stack_voltage_v: float
    internal_stack_temperature_k: float


def model_outputs_from_trajectory_point(
    point: V11TrajectoryPoint,
    *,
    stack: UserStackConfiguration | None = None,
) -> V11ModelOutputs:
    """Build the model-output vector from one predictive trajectory sample."""
    diagnostics = point.diagnostics
    if diagnostics.voltage is None:
        raise ValueError("predictive trajectory requires voltage diagnostics")
    if diagnostics.heat_transfer is None:
        raise ValueError("predictive trajectory requires heat-transfer diagnostics")

    cfg = UserStackConfiguration() if stack is None else stack
    cell_voltage = diagnostics.voltage.cell_voltage_v
    return V11ModelOutputs(
        stack_temperature_k=point.state.stack_temperature_k,
        cell_voltage_v=cell_voltage,
        stack_voltage_v=cfg.stack_voltage_v(cell_voltage),
        air_outlet_temperature_k=diagnostics.heat_transfer.outlet_temperature_k,
        anode_water_activity=diagnostics.anode_water_activity,
        cathode_water_activity=diagnostics.cathode_water_activity,
    )


def measurements_from_trajectory_point(
    point: V11TrajectoryPoint,
    *,
    stack: UserStackConfiguration | None = None,
) -> V11Measurements:
    """Map one model sample onto the three physical sensor channels."""
    outputs = model_outputs_from_trajectory_point(point, stack=stack)
    return V11Measurements(
        current_a=point.current_a,
        stack_voltage_v=outputs.stack_voltage_v,
        internal_stack_temperature_k=point.state.stack_temperature_k,
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
    measurements: V11Measurements


def state_space_sample_from_trajectory_point(
    point: V11TrajectoryPoint,
    inputs: V11RunnerInputs,
    *,
    stack: UserStackConfiguration | None = None,
) -> V11StateSpaceSample:
    """Expose one runner sample as (t, x, u, d, p, y_model, y_meas)."""
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
    outputs = model_outputs_from_trajectory_point(point, stack=stack)
    measurements = measurements_from_trajectory_point(point, stack=stack)
    return V11StateSpaceSample(
        time_s=point.time_s,
        state=point.state,
        manipulated=manipulated,
        exogenous=exogenous,
        parameters=parameters,
        outputs=outputs,
        measurements=measurements,
    )
