"""Print the nominal V11 local observability report as JSON."""

from __future__ import annotations

import json

from pemfc_dedalus.v11_interfaces import V11ExogenousInputs, V11ModelParameters
from pemfc_dedalus.v11_observability import local_observability_report
from pemfc_dedalus.v11_system import V11DynamicState


def main() -> None:
    state = V11DynamicState(
        stack_temperature_k=313.15,
        membrane_mean_water_content=5.0,
        anode_hydrogen_mol=4.5e-4,
        anode_nitrogen_mol=2.0e-5,
        anode_water_vapour_mol=1.0e-5,
        cathode_oxygen_mol=2.1e-4,
        cathode_nitrogen_mol=7.9e-4,
        cathode_total_water_mol=5.0e-5,
    )
    exogenous = V11ExogenousInputs(
        current_a=26.04,
        inlet_air_temperature_k=293.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.01,
    )
    parameters = V11ModelParameters(
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )
    report = local_observability_report(
        state=state,
        current_a=26.04,
        stack_air_flow_slpm=216.1,
        exogenous=exogenous,
        parameters=parameters,
    )
    print(
        json.dumps(
            {
                "state_names": report.state_names,
                "rank": report.rank,
                "state_dimension": report.state_dimension,
                "full_rank": report.full_rank,
                "singular_values": report.singular_values,
                "relative_tolerance": report.relative_tolerance,
                "current_sensitivity": report.current_measurement_state_sensitivity,
                "voltage_sensitivity": report.voltage_state_sensitivity,
                "temperature_sensitivity": report.temperature_state_sensitivity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
