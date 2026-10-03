"""Tests for V11 supervisory digital-twin controls."""

from __future__ import annotations

import pytest

from pemfc_dedalus.ballard_1020acs import (
    Ballard1020ACSTechnologyReference,
    UserStackConfiguration,
)
from pemfc_dedalus.v11_control import (
    minimum_recommended_airflow_slpm,
    supervisory_commands_at_time,
    supervisory_control_decision,
)
from pemfc_dedalus.v11_system import V11DynamicState


def _state(
    *,
    temperature_k: float = 313.15,
    n2_mol: float = 2.0e-5,
    water_mol: float = 1.0e-5,
) -> V11DynamicState:
    return V11DynamicState(
        stack_temperature_k=temperature_k,
        membrane_mean_water_content=5.0,
        anode_hydrogen_mol=4.5e-4,
        anode_nitrogen_mol=n2_mol,
        anode_water_vapour_mol=water_mol,
        cathode_oxygen_mol=2.1e-4,
        cathode_nitrogen_mol=7.9e-4,
        cathode_total_water_mol=5.0e-5,
    )


def test_minimum_airflow_uses_ballard_recommended_stoichiometry() -> None:
    current = 26.04
    stack = UserStackConfiguration()
    tech = Ballard1020ACSTechnologyReference()

    airflow = minimum_recommended_airflow_slpm(current)

    assert airflow == pytest.approx(
        stack.stoichiometric_air_slpm(current)
        * tech.oxidant_stoich_min_recommended
    )


def test_supervisory_airflow_preserves_feedforward_below_target() -> None:
    current = 26.04
    tech = Ballard1020ACSTechnologyReference()
    target = 273.15 + tech.optimum_stack_temperature_c(current)

    decision = supervisory_control_decision(
        state=_state(temperature_k=target - 1.0),
        current_a=current,
        steady_air_flow_slpm=216.1,
        maximum_air_flow_slpm=300.0,
        max_anode_nitrogen_mole_fraction=0.20,
        max_anode_water_mole_fraction=0.20,
    )

    assert decision.stack_air_flow_slpm == pytest.approx(216.1)
    assert not decision.purge_command


def test_supervisory_airflow_reaches_actuator_limit_at_temperature_limit() -> None:
    tech = Ballard1020ACSTechnologyReference()

    decision = supervisory_control_decision(
        state=_state(temperature_k=273.15 + tech.oxidant_temp_max_c),
        current_a=26.04,
        steady_air_flow_slpm=216.1,
        maximum_air_flow_slpm=300.0,
        max_anode_nitrogen_mole_fraction=0.20,
        max_anode_water_mole_fraction=0.20,
    )

    assert decision.stack_air_flow_slpm == pytest.approx(300.0)


def test_supervisory_purge_can_be_triggered_by_nitrogen_state() -> None:
    decision = supervisory_control_decision(
        state=_state(n2_mol=2.0e-4),
        current_a=26.04,
        steady_air_flow_slpm=216.1,
        maximum_air_flow_slpm=300.0,
        max_anode_nitrogen_mole_fraction=0.20,
        max_anode_water_mole_fraction=0.50,
    )

    assert decision.anode_nitrogen_mole_fraction >= 0.20
    assert decision.purge_command


def test_supervisory_purge_can_be_triggered_by_water_state() -> None:
    decision = supervisory_control_decision(
        state=_state(water_mol=2.0e-4),
        current_a=26.04,
        steady_air_flow_slpm=216.1,
        maximum_air_flow_slpm=300.0,
        max_anode_nitrogen_mole_fraction=0.50,
        max_anode_water_mole_fraction=0.20,
    )

    assert decision.anode_water_mole_fraction >= 0.20
    assert decision.purge_command


def test_supervisory_controller_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        supervisory_control_decision(
            state=_state(),
            current_a=26.04,
            steady_air_flow_slpm=216.1,
            maximum_air_flow_slpm=300.0,
            max_anode_nitrogen_mole_fraction=1.0,
            max_anode_water_mole_fraction=0.20,
        )


def test_supervisory_decision_translates_to_runner_commands() -> None:
    control, purge_commands, decision = supervisory_commands_at_time(
        time_s=1.0,
        state=_state(n2_mol=2.0e-4),
        current_a=26.04,
        steady_air_flow_slpm=216.1,
        maximum_air_flow_slpm=300.0,
        max_anode_nitrogen_mole_fraction=0.20,
        max_anode_water_mole_fraction=0.50,
    )

    assert control.start_time_s == pytest.approx(1.0)
    assert control.current_a == pytest.approx(26.04)
    assert control.stack_air_flow_slpm == pytest.approx(
        decision.stack_air_flow_slpm
    )
    assert len(purge_commands) == 1
    assert purge_commands[0].time_s == pytest.approx(1.0)
