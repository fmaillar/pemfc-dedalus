"""Map V11 local observability across current and purge state."""

from __future__ import annotations

import json

from pemfc_dedalus.ballard_1020acs import UserStackConfiguration
from pemfc_dedalus.v11_interfaces import V11ExogenousInputs, V11ModelParameters
from pemfc_dedalus.v11_observability import local_observability_report
from pemfc_dedalus.v11_purge import apply_v11_purge_event
from pemfc_dedalus.v11_system import V11DynamicState


def _pre_purge_state() -> V11DynamicState:
    return V11DynamicState(
        stack_temperature_k=313.15,
        membrane_mean_water_content=5.0,
        anode_hydrogen_mol=4.5e-4,
        anode_nitrogen_mol=2.0e-5,
        anode_water_vapour_mol=1.0e-5,
        cathode_oxygen_mol=2.1e-4,
        cathode_nitrogen_mol=7.9e-4,
        cathode_total_water_mol=5.0e-5,
    )


def main() -> None:
    stack = UserStackConfiguration()
    pre = _pre_purge_state()
    post = apply_v11_purge_event(state=pre).state
    states = {
        "pre_purge": pre,
        "post_purge": post,
    }
    currents = (5.0, 10.0, 15.0, 26.04)
    exogenous = V11ExogenousInputs(
        current_a=0.0,
        inlet_air_temperature_k=293.15,
        cathode_total_pressure_pa=101325.0,
        inlet_oxygen_mole_fraction=0.2095,
        inlet_water_mole_fraction=0.01,
    )
    parameters = V11ModelParameters(
        cathode_platinum_loading_mg_cm2_geo=0.4,
        cathode_ecsa_m2_pt_g_pt=50.0,
    )

    rows = []
    for state_label, state in states.items():
        for current_a in currents:
            airflow = stack.cathode_air_target_slpm(current_a)
            report = local_observability_report(
                state=state,
                current_a=current_a,
                stack_air_flow_slpm=airflow,
                exogenous=exogenous,
                parameters=parameters,
            )
            rows.append(
                {
                    "state": state_label,
                    "current_a": current_a,
                    "airflow_slpm": airflow,
                    "rank": report.rank,
                    "state_dimension": report.state_dimension,
                    "singular_values": report.singular_values,
                    "sigma_min": report.singular_values[-1],
                    "sigma_max": report.singular_values[0],
                    "condition_indicator": (
                        report.singular_values[0] / report.singular_values[-1]
                    ),
                }
            )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
