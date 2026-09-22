"""Manual-derived reference data for the Ballard FCgen/FCvelocity 1020ACS.

Source: Ballard MAN5100319-0B, June 20, 2014.

Only stack/system-level quantities explicitly documented in the manual are
included here. Internal active area and proprietary MEA/GDL/CL geometry are not
inferred from envelope dimensions.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ballard1020ACSReference:
    # Stack configuration
    n_cells: int = 56
    width_m: float = 0.351
    length_m: float = 0.103

    # The manual's dimension figure contains an OCR ambiguity in one line;
    # the dimension formula consistent with the figure and prior manual review
    # is kept explicit here.
    cell_pitch_m: float = 5.5e-3
    end_hardware_height_m: float = 55.4e-3

    # Electrical limits
    current_min_a: float = 0.0
    current_max_a: float = 75.0
    vcell_min_v: float = 0.5
    vcell_max_v: float = 1.0
    bus_plate_resistance_ohm: float = 2.2e-3

    # Typical BOL polarization curve (points above 75 A are reference-only).
    bol_current_a: tuple[float, ...] = (
        0.0, 7.3, 14.5, 29.0, 51.7, 65.3, 77.0, 87.1
    )
    bol_vcell_typ_v: tuple[float, ...] = (
        0.99, 0.83, 0.80, 0.76, 0.70, 0.66, 0.62, 0.58
    )
    bol_vcell_min_v: tuple[float, ...] = (
        0.93, 0.80, 0.77, 0.73, 0.67, 0.63, 0.58, 0.52
    )

    # Open-cathode / coolant-air constraints
    oxidant_temp_min_c: float = -20.0
    oxidant_temp_max_c: float = 52.0
    oxidant_temp_opt_min_c: float = 10.0
    oxidant_temp_opt_max_c: float = 40.0
    oxidant_rh_min: float = 0.0
    oxidant_rh_max: float = 1.0
    oxidant_stoich_no_loss: float = 50.0
    oxidant_stoich_min_recommended: float = 20.0

    # Dead-end H2 anode
    h2_pressure_min_barg: float = 0.16
    h2_pressure_opt_barg: float = 0.36
    h2_pressure_max_barg: float = 0.56
    h2_inlet_rh: float = 0.0
    h2_dead_end_stoich: float = 1.0
    h2_stoich_including_purge: float = 1.07

    # Runtime purge strategy
    purge_volume_per_cell_m3: float = 20e-6
    purge_interval_as: float = 2300.0
    purge_duration_max_s: float = 0.5
    lab_purge_duration_s: float = 0.2
    purge_rate_min_slpm_per_cell: float = 2.4
    h2_utilization_with_standard_purge: float = 0.93

    @property
    def stack_height_m(self) -> float:
        return self.n_cells * self.cell_pitch_m + self.end_hardware_height_m

    @property
    def stack_power_typical_w(self) -> tuple[float, ...]:
        return tuple(
            self.n_cells * i * v
            for i, v in zip(self.bol_current_a, self.bol_vcell_typ_v)
        )

    @staticmethod
    def optimum_stack_temperature_c(current_a: float) -> float:
        """Normal-operation optimum stack temperature from manual fit."""
        return 26.01 + 0.53 * current_a

    def heat_rejection_w(self, current_a: float, vcell_v: float) -> float:
        """Approximate stack heat generation used by Ballard manual."""
        return self.n_cells * current_a * (1.253 - vcell_v)

    def purge_period_s(self, current_a: float) -> float:
        """Runtime purge interval at constant current."""
        if current_a <= 0:
            return float("inf")
        return self.purge_interval_as / current_a

    def reacted_h2_slpm(self, current_a: float) -> float:
        """Total H2 reacted by the stack at standard conditions."""
        return 0.00696 * current_a * self.n_cells

    def stoichiometric_air_slpm(self, current_a: float) -> float:
        """Air required at stoichiometry 1 from the manual appendix."""
        return 0.0166 * current_a * self.n_cells
