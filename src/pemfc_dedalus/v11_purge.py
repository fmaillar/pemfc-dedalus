"""Purge-event controller for the V11 dead-end anode model.

The Ballard purge rule is represented as a charge-integrating event clock.
Purge transport itself remains the V11 perfectly mixed displacement model:
20 mL/cell are exchanged, corresponding to two anode gas volumes.

This module does not smear purge over the continuous RHS. It reports exact
event timing inside a numerical step so a time integrator can split the step
at the purge boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .ballard_1020acs import Ballard1020ACSTechnologyReference
from .v11_anode import (
    AnodeGasState,
    PurgeResult,
    apply_well_mixed_purge,
    hydrogen_moles_for_target_pressure,
)
from .v11_system import V11DynamicState


@dataclass(frozen=True)
class V11PurgeClock:
    """Accumulated electrical charge since the last purge [A s]."""

    charge_since_purge_as: float = 0.0


@dataclass(frozen=True)
class V11PurgeStep:
    """Purge-clock result over one proposed integration interval."""

    event_occurs: bool
    event_offset_s: float | None
    charge_before_as: float
    charge_at_end_without_event_as: float
    charge_after_event_as: float


@dataclass(frozen=True)
class V11PurgeEvent:
    """Dynamic-state reset and purge diagnostics."""

    state: V11DynamicState
    anode: PurgeResult
    exchange_volume_m3: float
    hydrogen_refill_mol: float
    target_pressure_pa: float


def time_to_ballard_purge_s(
    *,
    clock: V11PurgeClock,
    current_a: float,
    purge_interval_as: float = 2300.0,
) -> float:
    """Return time until the next charge-triggered purge."""
    if clock.charge_since_purge_as < 0.0:
        raise ValueError("charge_since_purge_as must be non-negative")
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if purge_interval_as <= 0.0:
        raise ValueError("purge_interval_as must be positive")
    if clock.charge_since_purge_as >= purge_interval_as:
        return 0.0
    if current_a == 0.0:
        return math.inf
    return (
        purge_interval_as - clock.charge_since_purge_as
    ) / current_a


def advance_ballard_purge_clock(
    *,
    clock: V11PurgeClock,
    current_a: float,
    dt_s: float,
    purge_interval_as: float = 2300.0,
) -> V11PurgeStep:
    """Advance the charge clock and locate at most one purge inside the step.

    The function intentionally requires dt to contain at most one purge event.
    A dynamic integrator should split larger intervals at each returned event.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if current_a < 0.0:
        raise ValueError("current_a must be non-negative")
    if purge_interval_as <= 0.0:
        raise ValueError("purge_interval_as must be positive")
    if clock.charge_since_purge_as < 0.0:
        raise ValueError("charge_since_purge_as must be non-negative")
    if clock.charge_since_purge_as >= purge_interval_as:
        raise ValueError(
            "clock must be reset before advancing past a purge threshold"
        )

    accumulated = clock.charge_since_purge_as + current_a * dt_s
    if accumulated < purge_interval_as:
        return V11PurgeStep(
            event_occurs=False,
            event_offset_s=None,
            charge_before_as=clock.charge_since_purge_as,
            charge_at_end_without_event_as=accumulated,
            charge_after_event_as=accumulated,
        )

    if current_a == 0.0:
        raise RuntimeError("zero current cannot cross the purge threshold")

    event_offset = (
        purge_interval_as - clock.charge_since_purge_as
    ) / current_a
    remaining_time = dt_s - event_offset
    charge_after_event = current_a * remaining_time
    if charge_after_event >= purge_interval_as:
        raise ValueError(
            "dt_s spans more than one purge event; split the integration step"
        )

    return V11PurgeStep(
        event_occurs=True,
        event_offset_s=event_offset,
        charge_before_as=clock.charge_since_purge_as,
        charge_at_end_without_event_as=accumulated,
        charge_after_event_as=charge_after_event,
    )


def clock_after_purge_step(step: V11PurgeStep) -> V11PurgeClock:
    """Return the purge clock after applying the event, if any."""
    return V11PurgeClock(charge_since_purge_as=step.charge_after_event_as)


def apply_v11_purge_event(
    *,
    state: V11DynamicState,
    technology: Ballard1020ACSTechnologyReference | None = None,
    purge_exchange_volume_m3: float | None = None,
) -> V11PurgeEvent:
    """Apply one well-mixed purge reset to the V11 dynamic state.

    By default the exchanged volume is the Ballard run-time purge volume of
    20 mL/cell. The displaced mixed gas is immediately replaced by regulated
    hydrogen to recover the Ballard nominal anode pressure. Thermal, membrane
    and cathode states remain continuous.
    """
    tech = (
        Ballard1020ACSTechnologyReference()
        if technology is None
        else technology
    )
    exchange_volume = (
        tech.purge_volume_per_cell_m3
        if purge_exchange_volume_m3 is None
        else purge_exchange_volume_m3
    )
    if exchange_volume < 0.0:
        raise ValueError("purge_exchange_volume_m3 must be non-negative")

    anode_state = AnodeGasState(
        hydrogen_mol=state.anode_hydrogen_mol,
        nitrogen_mol=state.anode_nitrogen_mol,
        water_vapour_mol=state.anode_water_vapour_mol,
    )
    purge = apply_well_mixed_purge(
        state=anode_state,
        purge_exchange_volume_m3=exchange_volume,
        anode_gas_volume_m3=tech.anode_gas_volume_per_cell_m3,
    )

    target_pressure_pa = 101325.0 + tech.h2_pressure_opt_barg * 1.0e5
    refilled_hydrogen = hydrogen_moles_for_target_pressure(
        target_total_pressure_pa=target_pressure_pa,
        nitrogen_mol=purge.state.nitrogen_mol,
        water_vapour_mol=purge.state.water_vapour_mol,
        volume_m3=tech.anode_gas_volume_per_cell_m3,
        temperature_k=state.stack_temperature_k,
    )
    final_hydrogen = max(
        purge.state.hydrogen_mol,
        refilled_hydrogen,
    )
    hydrogen_refill = final_hydrogen - purge.state.hydrogen_mol
    updated = replace(
        state,
        anode_hydrogen_mol=final_hydrogen,
        anode_nitrogen_mol=purge.state.nitrogen_mol,
        anode_water_vapour_mol=purge.state.water_vapour_mol,
    )
    return V11PurgeEvent(
        state=updated,
        anode=purge,
        exchange_volume_m3=exchange_volume,
        hydrogen_refill_mol=hydrogen_refill,
        target_pressure_pa=target_pressure_pa,
    )
