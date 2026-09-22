"""Manual-derived Ballard FCgen/FCvelocity 1020ACS technology reference.

Source: Ballard MAN5100319-0B, June 20, 2014.

This module contains technology/family-level data from the manual.  It does NOT
assume a 56-cell stack.  The user's physical stack is represented separately by
UserStackConfiguration.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ballard1020ACSTechnologyReference:
    """1020ACS family-level operating data from the Ballard manual."""

    # Envelope dimensions given by the product manual.
    width_m: float = 0.351
    length_m: float = 0.103

    # Cell pitch / end hardware relation shown in the stack dimension figure.
    cell_pitch_m: float = 5.5e-3
    end_hardware_height_m: float = 55.4e-3

    # Electrical operating limits.
    current_min_a: float = 0.0
    current_max_a: float = 75.0
    vcell_min_v: float = 0.5
    vcell_max_v: float = 1.0
    bus_plate_resistance_ohm: float = 2.2e-3

    # Typical BOL per-cell polarization data from the manual.
    bol_current_a: tuple[float, ...] = (
        0.0, 7.3, 14.5, 29.0, 51.7, 65.3, 77.0, 87.1
    )
    bol_vcell_typ_v: tuple[float, ...] = (
        0.99, 0.83, 0.80, 0.76, 0.70, 0.66, 0.62, 0.58
    )
    bol_vcell_min_v: tuple[float, ...] = (
        0.93, 0.80, 0.77, 0.73, 0.67, 0.63, 0.58, 0.52
    )

    # Open-cathode oxidant / coolant air.
    oxidant_temp_min_c: float = -20.0
    oxidant_temp_max_c: float = 52.0
    oxidant_temp_opt_min_c: float = 10.0
    oxidant_temp_opt_max_c: float = 40.0
    oxidant_rh_min: float = 0.0
    oxidant_rh_max: float = 1.0
    oxidant_stoich_no_loss: float = 50.0
    oxidant_stoich_min_recommended: float = 20.0

    # Dead-end H2 anode.
    h2_pressure_min_barg: float = 0.16
    h2_pressure_opt_barg: float = 0.36
    h2_pressure_max_barg: float = 0.56
    h2_inlet_rh: float = 0.0
    h2_dead_end_stoich: float = 1.0
    h2_stoich_including_purge: float = 1.07

    # Runtime purge.
    purge_volume_per_cell_m3: float = 20e-6
    purge_interval_as: float = 2300.0
    purge_duration_max_s: float = 0.5
    lab_purge_duration_s: float = 0.2
    purge_rate_min_slpm_per_cell: float = 2.4
    h2_utilization_with_standard_purge: float = 0.93

    @staticmethod
    def optimum_stack_temperature_c(current_a: float) -> float:
        return 26.01 + 0.53 * current_a

    @staticmethod
    def reacted_h2_slpm_per_cell(current_a: float) -> float:
        return 0.00696 * current_a

    @staticmethod
    def stoichiometric_air_slpm_per_cell(current_a: float) -> float:
        return 0.0166 * current_a


@dataclass(frozen=True)
class UserStackConfiguration:
    """Configuration of the user's actual nominal-200-W stack.

    The cell count is provisional and intentionally explicit.  It can be changed
    once the stack is counted or its nameplate / drawing is available.
    """

    n_cells: int = 10
    rated_power_w: float = 200.0

    @property
    def nominal_stack_height_m(self) -> float:
        tech = Ballard1020ACSTechnologyReference()
        return self.n_cells * tech.cell_pitch_m + tech.end_hardware_height_m

    @property
    def nominal_power_per_cell_w(self) -> float:
        return self.rated_power_w / self.n_cells

    def stack_voltage_v(self, vcell_v: float) -> float:
        return self.n_cells * vcell_v

    def stack_power_w(self, current_a: float, vcell_v: float) -> float:
        return self.n_cells * current_a * vcell_v

    def heat_rejection_w(self, current_a: float, vcell_v: float) -> float:
        return self.n_cells * current_a * (1.253 - vcell_v)

    def purge_period_s(self, current_a: float) -> float:
        tech = Ballard1020ACSTechnologyReference()
        if current_a <= 0:
            return float("inf")
        return tech.purge_interval_as / current_a

    def purge_volume_m3(self) -> float:
        tech = Ballard1020ACSTechnologyReference()
        return self.n_cells * tech.purge_volume_per_cell_m3

    def reacted_h2_slpm(self, current_a: float) -> float:
        tech = Ballard1020ACSTechnologyReference()
        return self.n_cells * tech.reacted_h2_slpm_per_cell(current_a)

    def stoichiometric_air_slpm(self, current_a: float) -> float:
        tech = Ballard1020ACSTechnologyReference()
        return self.n_cells * tech.stoichiometric_air_slpm_per_cell(current_a)

    def coolant_air_target_slpm(
        self,
        current_a: float,
        stoich: float | None = None,
    ) -> float:
        """Minimum oxidant-side air target before the future thermal controller.

        Cooling generally requires much more air than oxygen stoichiometry; for
        now use the manual's no-loss stoichiometry as a conservative floor.
        """
        tech = Ballard1020ACSTechnologyReference()
        air_stoich = tech.oxidant_stoich_no_loss if stoich is None else stoich
        return self.stoichiometric_air_slpm(current_a) * air_stoich
