"""State-space interface for the V11 galvanostatic PEMFC digital twin.

The dynamic state itself remains V11DynamicState. This module only classifies
model quantities into manipulated inputs, exogenous disturbances, model
parameters and measurable outputs. It contains no control law.
"""

from __future__ import annotations

from dataclasses import dataclass

from .v11_runner import V11TrajectoryPoint


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
class V11MeasuredOutputs:
    """Quantities that can be exposed as the model measurement vector y."""

    current_a: float
    stack_air_flow_slpm: float
    purge_event: bool
    stack_temperature_k: float
    cell_voltage_v: float
    air_outlet_temperature_k: float
    anode_water_activity: float
    cathode_water_activity: float


def measured_outputs_from_trajectory_point(
    point: V11TrajectoryPoint,
) -> V11MeasuredOutputs:
    """Build the measurement vector from one predictive trajectory sample."""
    diagnostics = point.diagnostics
    if diagnostics.voltage is None:
        raise ValueError("predictive trajectory requires voltage diagnostics")
    if diagnostics.heat_transfer is None:
        raise ValueError("predictive trajectory requires heat-transfer diagnostics")

    return V11MeasuredOutputs(
        current_a=point.current_a,
        stack_air_flow_slpm=point.stack_air_flow_slpm,
        purge_event=point.purge_event,
        stack_temperature_k=point.state.stack_temperature_k,
        cell_voltage_v=diagnostics.voltage.cell_voltage_v,
        air_outlet_temperature_k=diagnostics.heat_transfer.outlet_temperature_k,
        anode_water_activity=diagnostics.anode_water_activity,
        cathode_water_activity=diagnostics.cathode_water_activity,
    )
