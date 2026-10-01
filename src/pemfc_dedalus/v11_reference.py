"""Reference initialisation and control protocol for V11 dynamic studies.

The helpers in this module define one reproducible, explicitly assumed dynamic
experiment. They are not fitted operating data and are not used by the V11
constitutive closures themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anode import water_saturation_pressure_pa
from .ballard_1020acs import Ballard1020ACSTechnologyReference
from .membrane import membrane_water_content_from_activity
from .v11_heat_transfer import V11CathodeChannelGeometry
from .v11_runner import V11ControlSegment, V11RunnerInputs
from .v11_system import V11DynamicState

GAS_CONSTANT_J_MOL_K = 8.31446261815324


@dataclass(frozen=True)
class V11ReferenceScenario:
    """Fully specified reference transient excluding numerical time step."""

    initial_state: V11DynamicState
    controls: tuple[V11ControlSegment, ...]
    inputs: V11RunnerInputs
    stop_time_s: float


def humid_air_mole_fractions(
    *,
    temperature_k: float,
    total_pressure_pa: float,
    relative_humidity: float,
    dry_oxygen_mole_fraction: float = 0.2095,
) -> tuple[float, float, float]:
    """Return wet (O2, N2, H2O) mole fractions for humidified dry air."""
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if total_pressure_pa <= 0.0:
        raise ValueError("total_pressure_pa must be positive")
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be in [0, 1]")
    if not 0.0 < dry_oxygen_mole_fraction < 1.0:
        raise ValueError("dry_oxygen_mole_fraction must lie in (0, 1)")

    water_fraction = (
        relative_humidity
        * water_saturation_pressure_pa(temperature_k)
        / total_pressure_pa
    )
    if water_fraction >= 1.0:
        raise ValueError("humid-air water mole fraction must be below one")
    dry_fraction = 1.0 - water_fraction
    oxygen_fraction = dry_fraction * dry_oxygen_mole_fraction
    nitrogen_fraction = dry_fraction * (1.0 - dry_oxygen_mole_fraction)
    return oxygen_fraction, nitrogen_fraction, water_fraction


def reference_initial_state(
    *,
    stack_temperature_k: float = 303.15,
    cathode_pressure_pa: float = 101325.0,
    cathode_relative_humidity: float = 0.50,
    membrane_preconditioning_activity: float = 0.50,
) -> V11DynamicState:
    """Build a geometry- and pressure-consistent reference V11 state.

    The cathode gas inventory uses the published 80-channel flow volume. The
    anode starts with dry regulated hydrogen at the Ballard nominal 0.36 barg.
    Membrane hydration is an explicit 50% activity preconditioning assumption.
    """
    if stack_temperature_k <= 0.0:
        raise ValueError("stack_temperature_k must be positive")

    tech = Ballard1020ACSTechnologyReference()
    cathode_geometry = V11CathodeChannelGeometry()
    oxygen_fraction, nitrogen_fraction, water_fraction = (
        humid_air_mole_fractions(
            temperature_k=stack_temperature_k,
            total_pressure_pa=cathode_pressure_pa,
            relative_humidity=cathode_relative_humidity,
        )
    )
    cathode_total_mol = (
        cathode_pressure_pa
        * cathode_geometry.gas_volume_per_cell_m3
        / (GAS_CONSTANT_J_MOL_K * stack_temperature_k)
    )

    anode_target_pressure_pa = (
        101325.0 + tech.h2_pressure_opt_barg * 1.0e5
    )
    anode_hydrogen_mol = (
        anode_target_pressure_pa
        * tech.anode_gas_volume_per_cell_m3
        / (GAS_CONSTANT_J_MOL_K * stack_temperature_k)
    )
    membrane_water_content = float(
        membrane_water_content_from_activity(
            membrane_preconditioning_activity
        )
    )

    return V11DynamicState(
        stack_temperature_k=stack_temperature_k,
        membrane_mean_water_content=membrane_water_content,
        anode_hydrogen_mol=anode_hydrogen_mol,
        anode_nitrogen_mol=0.0,
        anode_water_vapour_mol=0.0,
        cathode_oxygen_mol=cathode_total_mol * oxygen_fraction,
        cathode_nitrogen_mol=cathode_total_mol * nitrogen_fraction,
        cathode_total_water_mol=cathode_total_mol * water_fraction,
    )


def reference_dynamic_scenario(
    *,
    cathode_platinum_loading_mg_cm2_geo: float = 0.4,
    cathode_ecsa_m2_pt_g_pt: float = 50.0,
) -> V11ReferenceScenario:
    """Return the first reproducible V11 current/ventilation transient.

    Pt loading and ECSA are deliberately explicit scenario inputs. The default
    values are generic sensitivity placeholders, not FCgen-1020ACS defaults.
    """
    inlet_temperature_k = 293.15
    cathode_pressure_pa = 101325.0
    inlet_relative_humidity = 0.50
    inlet_oxygen, _, inlet_water = humid_air_mole_fractions(
        temperature_k=inlet_temperature_k,
        total_pressure_pa=cathode_pressure_pa,
        relative_humidity=inlet_relative_humidity,
    )

    controls = (
        V11ControlSegment(0.0, 5.0, 100.0),
        V11ControlSegment(60.0, 15.0, 150.0),
        V11ControlSegment(180.0, 26.04, 216.1),
        V11ControlSegment(360.0, 10.0, 120.0),
    )
    inputs = V11RunnerInputs(
        inlet_air_temperature_k=inlet_temperature_k,
        cathode_total_pressure_pa=cathode_pressure_pa,
        inlet_oxygen_mole_fraction=inlet_oxygen,
        inlet_water_mole_fraction=inlet_water,
        cathode_platinum_loading_mg_cm2_geo=(
            cathode_platinum_loading_mg_cm2_geo
        ),
        cathode_ecsa_m2_pt_g_pt=cathode_ecsa_m2_pt_g_pt,
    )
    return V11ReferenceScenario(
        initial_state=reference_initial_state(),
        controls=controls,
        inputs=inputs,
        stop_time_s=480.0,
    )
