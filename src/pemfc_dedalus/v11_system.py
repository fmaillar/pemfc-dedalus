"""Coupled dynamic core for the V11 galvanostatic PEMFC model.

The core assembles the V11 cathode, dead-end anode, membrane hydration,
equilibrium cathode water partition and thermal balance.

The low-level RHS keeps externally supplied closures explicit for diagnostics.
The predictive RHS resolves cell voltage, cathode outlet molar flow, nitrogen
crossover and cathode air outlet temperature from the dynamic state and
published channel geometry.

No fitted fallback values are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .anode import water_saturation_pressure_pa
from .ballard_1020acs import (
    Ballard1020ACSTechnologyReference,
    UserStackConfiguration,
)
from .v11_anode import (
    AnodeGasDerivative,
    AnodeGasState,
    anode_gas_rhs_per_cell,
    ideal_gas_total_pressure_pa,
    regulator_hydrogen_inlet_mol_s,
)
from .v11_cathode import (
    CathodeGasDerivative,
    CathodeGasState,
    cathode_gas_rhs_per_cell,
    cathode_isobaric_outlet_mol_s,
    standard_litre_per_minute_to_mol_s,
)
from .v11_heat_transfer import (
    V11CathodeHeatTransfer,
    cathode_air_outlet_temperature_geometry_v11,
)
from .v11_membrane import (
    MembraneHydrationDerivative,
    ge_interface_water_flux_into_membrane,
    membrane_hydration_rhs,
)
from .v11_nitrogen import V11NitrogenCrossover, catalano_nitrogen_crossover
from .v11_phase_change import (
    CathodeWaterPhaseState,
    repartition_cathode_water_equilibrium,
)
from .v11_thermal import ThermalBalance, stack_temperature_rhs_k_s
from .v11_voltage import V11VoltagePrediction, predict_cell_voltage_v


@dataclass(frozen=True)
class V11DynamicState:
    """Reduced per-cell inventories plus lumped stack states."""

    stack_temperature_k: float
    membrane_mean_water_content: float
    anode_hydrogen_mol: float
    anode_nitrogen_mol: float
    anode_water_vapour_mol: float
    cathode_oxygen_mol: float
    cathode_nitrogen_mol: float
    cathode_total_water_mol: float


@dataclass(frozen=True)
class V11DynamicDerivative:
    """Time derivative of the reduced V11 state."""

    stack_temperature_k_s: float
    membrane_mean_water_content_s: float
    anode_hydrogen_mol_s: float
    anode_nitrogen_mol_s: float
    anode_water_vapour_mol_s: float
    cathode_oxygen_mol_s: float
    cathode_nitrogen_mol_s: float
    cathode_total_water_mol_s: float


@dataclass(frozen=True)
class V11CoupledDiagnostics:
    """Algebraic states and subsystem balances returned with the RHS."""

    anode_water_activity: float
    cathode_water_activity: float
    cathode_phase: CathodeWaterPhaseState
    membrane: MembraneHydrationDerivative
    anode: AnodeGasDerivative
    cathode: CathodeGasDerivative
    thermal: ThermalBalance
    hydrogen_inlet_mol_s: float
    cathode_water_inlet_mol_s: float
    cathode_water_outlet_mol_s: float
    water_conservation_residual_mol_s: float
    cathode_outlet_molar_flow_per_cell_mol_s: float
    voltage: V11VoltagePrediction | None = None
    nitrogen_crossover: V11NitrogenCrossover | None = None
    heat_transfer: V11CathodeHeatTransfer | None = None


def _bounded_water_activity(
    *,
    water_partial_pressure_pa: float,
    temperature_k: float,
) -> float:
    if water_partial_pressure_pa < 0.0:
        raise ValueError("water_partial_pressure_pa must be non-negative")
    saturation = water_saturation_pressure_pa(temperature_k)
    return min(max(water_partial_pressure_pa / saturation, 0.0), 1.0)


def coupled_v11_rhs(
    *,
    state: V11DynamicState,
    current_a: float,
    stack_air_flow_slpm: float,
    cathode_outlet_molar_flow_per_cell_mol_s: float,
    cell_voltage_v: float,
    inlet_air_temperature_k: float,
    cathode_air_outlet_temperature_k: float,
    cathode_total_pressure_pa: float,
    inlet_oxygen_mole_fraction: float,
    inlet_water_mole_fraction: float,
    nitrogen_crossover_mol_s: float,
    dt_regulator_s: float,
    stack: UserStackConfiguration | None = None,
    technology: Ballard1020ACSTechnologyReference | None = None,
    faraday_c_mol: float = 96485.33212,
) -> tuple[V11DynamicDerivative, V11CoupledDiagnostics]:
    """Return the coupled V11 RHS and diagnostics.

    Cathode liquid water is an algebraic equilibrium output derived from the
    total cathode-water inventory, not an independently integrated state.
    """
    cfg = UserStackConfiguration() if stack is None else stack
    tech = (
        Ballard1020ACSTechnologyReference()
        if technology is None
        else technology
    )

    if state.stack_temperature_k <= 0.0:
        raise ValueError("stack_temperature_k must be positive")
    if state.membrane_mean_water_content < 0.0:
        raise ValueError("membrane_mean_water_content must be non-negative")
    if cathode_total_pressure_pa <= 0.0:
        raise ValueError("cathode_total_pressure_pa must be positive")
    if nitrogen_crossover_mol_s < 0.0:
        raise ValueError("nitrogen_crossover_mol_s must be non-negative")
    if dt_regulator_s <= 0.0:
        raise ValueError("dt_regulator_s must be positive")

    cathode_phase = repartition_cathode_water_equilibrium(
        total_water_mol=state.cathode_total_water_mol,
        dry_gas_mol=state.cathode_oxygen_mol + state.cathode_nitrogen_mol,
        temperature_k=state.stack_temperature_k,
        total_pressure_pa=cathode_total_pressure_pa,
    )
    cathode_gas = CathodeGasState(
        oxygen_mol=state.cathode_oxygen_mol,
        nitrogen_mol=state.cathode_nitrogen_mol,
        water_vapour_mol=cathode_phase.vapour_mol,
    )
    _, _, cathode_water_fraction = cathode_gas.mole_fractions()
    cathode_water_activity = _bounded_water_activity(
        water_partial_pressure_pa=(
            cathode_water_fraction * cathode_total_pressure_pa
        ),
        temperature_k=state.stack_temperature_k,
    )

    anode_state = AnodeGasState(
        hydrogen_mol=state.anode_hydrogen_mol,
        nitrogen_mol=state.anode_nitrogen_mol,
        water_vapour_mol=state.anode_water_vapour_mol,
    )
    anode_pressure = ideal_gas_total_pressure_pa(
        state=anode_state,
        volume_m3=tech.anode_gas_volume_per_cell_m3,
        temperature_k=state.stack_temperature_k,
    )
    _, _, anode_water_fraction = anode_state.mole_fractions()
    anode_water_activity = _bounded_water_activity(
        water_partial_pressure_pa=anode_water_fraction * anode_pressure,
        temperature_k=state.stack_temperature_k,
    )

    anode_interface = ge_interface_water_flux_into_membrane(
        gas_water_activity=anode_water_activity,
        membrane_water_content=state.membrane_mean_water_content,
        temperature_k=state.stack_temperature_k,
    )
    cathode_interface = ge_interface_water_flux_into_membrane(
        gas_water_activity=cathode_water_activity,
        membrane_water_content=state.membrane_mean_water_content,
        temperature_k=state.stack_temperature_k,
    )
    membrane = membrane_hydration_rhs(
        anode_interface_flux_into_membrane_mol_m2_s=(
            anode_interface.flux_into_membrane_mol_m2_s
        ),
        cathode_interface_flux_into_membrane_mol_m2_s=(
            cathode_interface.flux_into_membrane_mol_m2_s
        ),
    )

    anode_water_source = -membrane.anode_interface_rate_mol_s
    target_anode_pressure_pa = (
        101325.0 + tech.h2_pressure_opt_barg * 1.0e5
    )
    hydrogen_inlet = regulator_hydrogen_inlet_mol_s(
        state=anode_state,
        target_total_pressure_pa=target_anode_pressure_pa,
        current_a=current_a,
        nitrogen_source_mol_s=nitrogen_crossover_mol_s,
        water_source_mol_s=anode_water_source,
        dt_s=dt_regulator_s,
        volume_m3=tech.anode_gas_volume_per_cell_m3,
        temperature_k=state.stack_temperature_k,
        faraday_c_mol=faraday_c_mol,
    )
    anode = anode_gas_rhs_per_cell(
        current_a=current_a,
        hydrogen_inlet_mol_s=hydrogen_inlet,
        nitrogen_source_mol_s=nitrogen_crossover_mol_s,
        water_source_mol_s=anode_water_source,
        faraday_c_mol=faraday_c_mol,
    )

    inlet_per_cell_mol_s = standard_litre_per_minute_to_mol_s(
        stack_air_flow_slpm / cfg.n_cells
    )
    cathode_water_inlet = inlet_per_cell_mol_s * inlet_water_mole_fraction
    cathode_water_outlet = (
        cathode_outlet_molar_flow_per_cell_mol_s * cathode_water_fraction
    )

    faraday_water = current_a / (2.0 * faraday_c_mol)

    cathode = cathode_gas_rhs_per_cell(
        state=cathode_gas,
        stack_air_flow_slpm=stack_air_flow_slpm,
        n_cells=cfg.n_cells,
        current_a=current_a,
        inlet_oxygen_mole_fraction=inlet_oxygen_mole_fraction,
        inlet_water_mole_fraction=inlet_water_mole_fraction,
        water_source_to_gas_mol_s=(
            faraday_water - membrane.cathode_interface_rate_mol_s
        ),
        outlet_molar_flow_per_cell_mol_s=(
            cathode_outlet_molar_flow_per_cell_mol_s
        ),
        nitrogen_sink_mol_s=nitrogen_crossover_mol_s,
        faraday_c_mol=faraday_c_mol,
    )

    # Water is evolved as a total cathode inventory so equilibrium phase
    # partition remains algebraic.
    cathode_total_water_rate = (
        faraday_water
        + cathode_water_inlet
        - cathode_water_outlet
        - membrane.cathode_interface_rate_mol_s
    )

    thermal = stack_temperature_rhs_k_s(
        n_cells=cfg.n_cells,
        current_a=current_a,
        cell_voltage_v=cell_voltage_v,
        stack_air_flow_slpm=stack_air_flow_slpm,
        inlet_temperature_k=inlet_air_temperature_k,
        outlet_temperature_k=cathode_air_outlet_temperature_k,
        thermal_mass_j_k_per_cell=tech.thermal_mass_j_k_per_cell,
    )

    total_water_storage_rate = (
        anode.water_vapour_mol_s
        + membrane.net_storage_rate_mol_s
        + cathode_total_water_rate
    )
    expected_external_water_rate = (
        faraday_water + cathode_water_inlet - cathode_water_outlet
    )
    water_residual = (
        total_water_storage_rate - expected_external_water_rate
    )

    derivative = V11DynamicDerivative(
        stack_temperature_k_s=thermal.temperature_rate_k_s,
        membrane_mean_water_content_s=membrane.mean_water_content_rate_s,
        anode_hydrogen_mol_s=anode.hydrogen_mol_s,
        anode_nitrogen_mol_s=anode.nitrogen_mol_s,
        anode_water_vapour_mol_s=anode.water_vapour_mol_s,
        cathode_oxygen_mol_s=cathode.oxygen_mol_s,
        cathode_nitrogen_mol_s=cathode.nitrogen_mol_s,
        cathode_total_water_mol_s=cathode_total_water_rate,
    )
    diagnostics = V11CoupledDiagnostics(
        anode_water_activity=anode_water_activity,
        cathode_water_activity=cathode_water_activity,
        cathode_phase=cathode_phase,
        membrane=membrane,
        anode=anode,
        cathode=cathode,
        thermal=thermal,
        hydrogen_inlet_mol_s=hydrogen_inlet,
        cathode_water_inlet_mol_s=cathode_water_inlet,
        cathode_water_outlet_mol_s=cathode_water_outlet,
        water_conservation_residual_mol_s=water_residual,
        cathode_outlet_molar_flow_per_cell_mol_s=(
            cathode_outlet_molar_flow_per_cell_mol_s
        ),
    )
    return derivative, diagnostics



def coupled_v11_predictive_rhs(
    *,
    state: V11DynamicState,
    current_a: float,
    stack_air_flow_slpm: float,
    inlet_air_temperature_k: float,
    cathode_total_pressure_pa: float,
    inlet_oxygen_mole_fraction: float,
    inlet_water_mole_fraction: float,
    dt_regulator_s: float,
    cathode_platinum_loading_mg_cm2_geo: float,
    cathode_ecsa_m2_pt_g_pt: float,
    additional_resolved_loss_v: float = 0.0,
    stack: UserStackConfiguration | None = None,
    technology: Ballard1020ACSTechnologyReference | None = None,
    faraday_c_mol: float = 96485.33212,
) -> tuple[V11DynamicDerivative, V11CoupledDiagnostics]:
    """Return the V11 RHS with cell voltage predicted from the dynamic state.

    Pt loading and ECSA remain mandatory because no defensible
    FCgen-1020ACS-specific roughness factor has been established. Cathode outlet
    molar flow is closed isobarically, nitrogen crossover is predicted from
    Catalano Nafion permeability, and cathode-air outlet temperature is derived
    from published FCgen-1020ACS channel geometry and a laminar Nusselt law.
    """
    tech = (
        Ballard1020ACSTechnologyReference()
        if technology is None
        else technology
    )

    cathode_phase = repartition_cathode_water_equilibrium(
        total_water_mol=state.cathode_total_water_mol,
        dry_gas_mol=state.cathode_oxygen_mol + state.cathode_nitrogen_mol,
        temperature_k=state.stack_temperature_k,
        total_pressure_pa=cathode_total_pressure_pa,
    )
    cathode_gas = CathodeGasState(
        oxygen_mol=state.cathode_oxygen_mol,
        nitrogen_mol=state.cathode_nitrogen_mol,
        water_vapour_mol=cathode_phase.vapour_mol,
    )
    (
        cathode_oxygen_fraction,
        cathode_nitrogen_fraction,
        cathode_water_fraction,
    ) = cathode_gas.mole_fractions()
    oxygen_partial_pressure_pa = (
        cathode_oxygen_fraction * cathode_total_pressure_pa
    )
    cathode_water_activity = _bounded_water_activity(
        water_partial_pressure_pa=(
            cathode_water_fraction * cathode_total_pressure_pa
        ),
        temperature_k=state.stack_temperature_k,
    )

    anode_state = AnodeGasState(
        hydrogen_mol=state.anode_hydrogen_mol,
        nitrogen_mol=state.anode_nitrogen_mol,
        water_vapour_mol=state.anode_water_vapour_mol,
    )
    anode_pressure_pa = ideal_gas_total_pressure_pa(
        state=anode_state,
        volume_m3=tech.anode_gas_volume_per_cell_m3,
        temperature_k=state.stack_temperature_k,
    )
    hydrogen_fraction, anode_nitrogen_fraction, _ = (
        anode_state.mole_fractions()
    )
    hydrogen_partial_pressure_pa = hydrogen_fraction * anode_pressure_pa
    nitrogen_crossover = catalano_nitrogen_crossover(
        membrane_mean_water_content=state.membrane_mean_water_content,
        temperature_k=state.stack_temperature_k,
        cathode_nitrogen_partial_pressure_pa=(
            cathode_nitrogen_fraction * cathode_total_pressure_pa
        ),
        anode_nitrogen_partial_pressure_pa=(
            anode_nitrogen_fraction * anode_pressure_pa
        ),
    )

    voltage = predict_cell_voltage_v(
        current_a=current_a,
        temperature_k=state.stack_temperature_k,
        hydrogen_partial_pressure_pa=hydrogen_partial_pressure_pa,
        oxygen_partial_pressure_pa=oxygen_partial_pressure_pa,
        water_activity=cathode_water_activity,
        membrane_mean_water_content=state.membrane_mean_water_content,
        cathode_platinum_loading_mg_cm2_geo=(
            cathode_platinum_loading_mg_cm2_geo
        ),
        cathode_ecsa_m2_pt_g_pt=cathode_ecsa_m2_pt_g_pt,
        additional_resolved_loss_v=additional_resolved_loss_v,
    )

    cfg = UserStackConfiguration() if stack is None else stack
    heat_transfer = cathode_air_outlet_temperature_geometry_v11(
        stack_temperature_k=state.stack_temperature_k,
        inlet_temperature_k=inlet_air_temperature_k,
        stack_air_flow_slpm=stack_air_flow_slpm,
        n_cells=cfg.n_cells,
    )
    thermal = stack_temperature_rhs_k_s(
        n_cells=cfg.n_cells,
        current_a=current_a,
        cell_voltage_v=voltage.cell_voltage_v,
        stack_air_flow_slpm=stack_air_flow_slpm,
        inlet_temperature_k=inlet_air_temperature_k,
        outlet_temperature_k=heat_transfer.outlet_temperature_k,
        thermal_mass_j_k_per_cell=tech.thermal_mass_j_k_per_cell,
    )
    inlet_per_cell_mol_s = standard_litre_per_minute_to_mol_s(
        stack_air_flow_slpm / cfg.n_cells
    )
    cathode_interface_rate = membrane_hydration_rhs(
        anode_interface_flux_into_membrane_mol_m2_s=0.0,
        cathode_interface_flux_into_membrane_mol_m2_s=(
            ge_interface_water_flux_into_membrane(
                gas_water_activity=cathode_water_activity,
                membrane_water_content=state.membrane_mean_water_content,
                temperature_k=state.stack_temperature_k,
            ).flux_into_membrane_mol_m2_s
        ),
    ).cathode_interface_rate_mol_s
    water_source_to_gas = (
        current_a / (2.0 * faraday_c_mol) - cathode_interface_rate
    )
    cathode_outlet_molar_flow_per_cell_mol_s = (
        cathode_isobaric_outlet_mol_s(
            state=cathode_gas,
            liquid_water_mol=cathode_phase.liquid_mol,
            inlet_air_mol_s=inlet_per_cell_mol_s,
            inlet_water_mole_fraction=inlet_water_mole_fraction,
            current_a=current_a,
            water_source_to_gas_mol_s=water_source_to_gas,
            nitrogen_sink_mol_s=nitrogen_crossover.rate_mol_s,
            temperature_k=state.stack_temperature_k,
            temperature_rate_k_s=thermal.temperature_rate_k_s,
            total_pressure_pa=cathode_total_pressure_pa,
            faraday_c_mol=faraday_c_mol,
        )
    )

    derivative, diagnostics = coupled_v11_rhs(
        state=state,
        current_a=current_a,
        stack_air_flow_slpm=stack_air_flow_slpm,
        cathode_outlet_molar_flow_per_cell_mol_s=(
            cathode_outlet_molar_flow_per_cell_mol_s
        ),
        cell_voltage_v=voltage.cell_voltage_v,
        inlet_air_temperature_k=inlet_air_temperature_k,
        cathode_air_outlet_temperature_k=heat_transfer.outlet_temperature_k,
        cathode_total_pressure_pa=cathode_total_pressure_pa,
        inlet_oxygen_mole_fraction=inlet_oxygen_mole_fraction,
        inlet_water_mole_fraction=inlet_water_mole_fraction,
        nitrogen_crossover_mol_s=nitrogen_crossover.rate_mol_s,
        dt_regulator_s=dt_regulator_s,
        stack=stack,
        technology=tech,
        faraday_c_mol=faraday_c_mol,
    )
    return derivative, replace(
        diagnostics,
        voltage=voltage,
        nitrogen_crossover=nitrogen_crossover,
        heat_transfer=heat_transfer,
    )
