"""Local observability analysis for the continuous V11 model.

The analysis linearises the continuous predictive RHS around one operating
point. Purge is a discrete hybrid reset and is intentionally excluded from the
continuous Jacobian. State scaling is applied before the observability rank is
computed because the V11 states span temperatures, hydration and molar
inventories with very different magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .ballard_1020acs import UserStackConfiguration
from .v11_interfaces import V11ExogenousInputs, V11ModelParameters
from .v11_system import V11DynamicState, coupled_v11_predictive_rhs

_STATE_NAMES = tuple(field.name for field in fields(V11DynamicState))


@dataclass(frozen=True)
class V11LocalObservabilityReport:
    """Numerical local observability result for one continuous operating point."""

    state_names: tuple[str, ...]
    rank: int
    state_dimension: int
    singular_values: tuple[float, ...]
    relative_tolerance: float
    current_measurement_state_sensitivity: tuple[float, ...]
    voltage_state_sensitivity: tuple[float, ...]
    temperature_state_sensitivity: tuple[float, ...]

    @property
    def full_rank(self) -> bool:
        return self.rank == self.state_dimension


def _state_vector(state: V11DynamicState) -> np.ndarray:
    return np.asarray(
        [getattr(state, name) for name in _STATE_NAMES],
        dtype=float,
    )


def _state_from_vector(vector: np.ndarray) -> V11DynamicState:
    return V11DynamicState(
        **{name: float(value) for name, value in zip(_STATE_NAMES, vector, strict=True)}
    )


def _state_scales(state: V11DynamicState) -> np.ndarray:
    values = np.abs(_state_vector(state))
    floors = np.asarray(
        [300.0, 1.0, 1.0e-4, 1.0e-7, 1.0e-7, 1.0e-4, 1.0e-4, 1.0e-7],
        dtype=float,
    )
    return np.maximum(values, floors)


def _rhs_vector(
    state: V11DynamicState,
    *,
    current_a: float,
    stack_air_flow_slpm: float,
    exogenous: V11ExogenousInputs,
    parameters: V11ModelParameters,
) -> np.ndarray:
    derivative, _ = coupled_v11_predictive_rhs(
        state=state,
        current_a=current_a,
        stack_air_flow_slpm=stack_air_flow_slpm,
        inlet_air_temperature_k=exogenous.inlet_air_temperature_k,
        cathode_total_pressure_pa=exogenous.cathode_total_pressure_pa,
        inlet_oxygen_mole_fraction=exogenous.inlet_oxygen_mole_fraction,
        inlet_water_mole_fraction=exogenous.inlet_water_mole_fraction,
        cathode_platinum_loading_mg_cm2_geo=(
            parameters.cathode_platinum_loading_mg_cm2_geo
        ),
        cathode_ecsa_m2_pt_g_pt=parameters.cathode_ecsa_m2_pt_g_pt,
        membrane_conductivity_multiplier=(
            parameters.membrane_conductivity_multiplier
        ),
        additional_resolved_loss_v=parameters.additional_resolved_loss_v,
    )
    return np.asarray(
        [getattr(derivative, name + "_s") for name in _STATE_NAMES],
        dtype=float,
    )


def _measurement_vector(
    state: V11DynamicState,
    *,
    current_a: float,
    stack_air_flow_slpm: float,
    exogenous: V11ExogenousInputs,
    parameters: V11ModelParameters,
    stack: UserStackConfiguration,
) -> np.ndarray:
    _, diagnostics = coupled_v11_predictive_rhs(
        state=state,
        current_a=current_a,
        stack_air_flow_slpm=stack_air_flow_slpm,
        inlet_air_temperature_k=exogenous.inlet_air_temperature_k,
        cathode_total_pressure_pa=exogenous.cathode_total_pressure_pa,
        inlet_oxygen_mole_fraction=exogenous.inlet_oxygen_mole_fraction,
        inlet_water_mole_fraction=exogenous.inlet_water_mole_fraction,
        cathode_platinum_loading_mg_cm2_geo=(
            parameters.cathode_platinum_loading_mg_cm2_geo
        ),
        cathode_ecsa_m2_pt_g_pt=parameters.cathode_ecsa_m2_pt_g_pt,
        membrane_conductivity_multiplier=(
            parameters.membrane_conductivity_multiplier
        ),
        additional_resolved_loss_v=parameters.additional_resolved_loss_v,
    )
    if diagnostics.voltage is None:
        raise ValueError("predictive RHS must provide voltage diagnostics")
    return np.asarray(
        [
            current_a,
            stack.stack_voltage_v(diagnostics.voltage.cell_voltage_v),
            state.stack_temperature_k,
        ],
        dtype=float,
    )


def _finite_difference_jacobian(
    function,
    state: V11DynamicState,
    *,
    relative_step: float,
) -> np.ndarray:
    base = _state_vector(state)
    scales = _state_scales(state)
    columns: list[np.ndarray] = []
    for index, scale in enumerate(scales):
        step = relative_step * scale
        plus = base.copy()
        minus = base.copy()
        plus[index] += step
        minus[index] -= step
        if minus[index] < 0.0:
            f0 = np.asarray(function(state), dtype=float)
            fplus = np.asarray(function(_state_from_vector(plus)), dtype=float)
            column = (fplus - f0) / step
        else:
            fplus = np.asarray(function(_state_from_vector(plus)), dtype=float)
            fminus = np.asarray(function(_state_from_vector(minus)), dtype=float)
            column = (fplus - fminus) / (2.0 * step)
        columns.append(column)
    return np.column_stack(columns)


def local_observability_report(
    *,
    state: V11DynamicState,
    current_a: float,
    stack_air_flow_slpm: float,
    exogenous: V11ExogenousInputs,
    parameters: V11ModelParameters,
    relative_step: float = 1.0e-6,
    relative_tolerance: float = 1.0e-8,
    stack: UserStackConfiguration | None = None,
) -> V11LocalObservabilityReport:
    """Return a scaled local observability rank for the continuous V11 model."""
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if stack_air_flow_slpm <= 0.0:
        raise ValueError("stack_air_flow_slpm must be positive")

    cfg = UserStackConfiguration() if stack is None else stack
    scales = _state_scales(state)

    a = _finite_difference_jacobian(
        lambda candidate: _rhs_vector(
            candidate,
            current_a=current_a,
            stack_air_flow_slpm=stack_air_flow_slpm,
            exogenous=exogenous,
            parameters=parameters,
        ),
        state,
        relative_step=relative_step,
    )
    c = _finite_difference_jacobian(
        lambda candidate: _measurement_vector(
            candidate,
            current_a=current_a,
            stack_air_flow_slpm=stack_air_flow_slpm,
            exogenous=exogenous,
            parameters=parameters,
            stack=cfg,
        ),
        state,
        relative_step=relative_step,
    )

    scale_matrix = np.diag(scales)
    inverse_scale_matrix = np.diag(1.0 / scales)
    a_scaled = inverse_scale_matrix @ a @ scale_matrix
    c_scaled = c @ scale_matrix

    blocks = [c_scaled]
    current_power = np.eye(len(_STATE_NAMES))
    for _ in range(1, len(_STATE_NAMES)):
        current_power = current_power @ a_scaled
        blocks.append(c_scaled @ current_power)
    observability = np.vstack(blocks)

    singular_values = np.linalg.svd(observability, compute_uv=False)
    threshold = relative_tolerance * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))

    return V11LocalObservabilityReport(
        state_names=_STATE_NAMES,
        rank=rank,
        state_dimension=len(_STATE_NAMES),
        singular_values=tuple(float(value) for value in singular_values),
        relative_tolerance=relative_tolerance,
        current_measurement_state_sensitivity=tuple(float(v) for v in c_scaled[0]),
        voltage_state_sensitivity=tuple(float(v) for v in c_scaled[1]),
        temperature_state_sensitivity=tuple(float(v) for v in c_scaled[2]),
    )
