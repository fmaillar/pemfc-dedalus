"""Physical and geometrical parameters for PEMFC validation models."""

from dataclasses import dataclass, field

from .ballard_1020acs import (
    Ballard1020ACSTechnologyReference,
    UserStackConfiguration,
)


@dataclass(frozen=True)
class CathodeParameters:
    """Parameters for the cathode GDL+CL models.

    System-level operating conditions are derived from the user's 200-W stack
    configuration where possible.  Internal MEA/GDL/CL geometry remains an
    explicit modelling assumption because the Ballard manual does not disclose it.
    """

    stack: UserStackConfiguration = field(default_factory=UserStackConfiguration)
    tech: Ballard1020ACSTechnologyReference = field(
        default_factory=Ballard1020ACSTechnologyReference
    )

    # Representative-cell model domain [m].
    # These are NOT Ballard envelope dimensions.
    length_x: float = 2.0e-3
    length_y: float = 1.0e-3
    gdl_thickness: float = 200e-6
    cl_thickness: float = 12e-6

    # Open-cathode ambient / gas state.
    # The oxidant inlet temperature is distinct from the stack/MEA temperature.
    # At the provisional 26.04 A nominal point, the manual fit gives
    # Topt = 26.01 + 0.53 I = 39.8112 degC.
    stack_temperature: float = 312.9612
    oxidant_inlet_temperature: float = 293.15
    pressure: float = 101325.0
    oxygen_mole_fraction: float = 0.2095
    relative_humidity: float = 0.50
    d_o2_bulk: float = 2.0e-5
    porosity_gdl: float = 0.70
    porosity_cl: float = 0.35

    # Operating point used by the validation model.
    stack_current_a: float = 26.04

    # V0.1 pseudo-first-order sink [1/s]
    k_reaction: float = 250.0

    # Electrochemistry, V0.2
    faraday: float = 96485.33212
    gas_constant: float = 8.31446261815324
    equilibrium_potential: float = 1.18
    cathode_solid_potential: float = 0.768
    membrane_proton_potential: float = 0.0
    alpha_anodic: float = 0.5
    alpha_cathodic: float = 0.5

    # Volumetric exchange-current scale [A/m^3].
    # Still a calibration parameter until active area / CL microstructure are known.
    j0_vol: float = 1.0e5
    oxygen_reaction_order: float = 1.0

    # Effective conductivities [S/m]
    sigma_s_gdl: float = 1.0e3
    sigma_s_cl: float = 3.0e2
    sigma_m_cl: float = 5.0
    sigma_m_floor: float = 1.0e-2

    # Pseudo-capacitances [F/m^3] for pseudo-transient convergence.
    pseudo_capacitance_s: float = 1.0e7
    pseudo_capacitance_m: float = 1.0e7

    interface_width: float = 2.0e-6

    # V0.3 hydrated-membrane validation assumptions.
    # Internal membrane data are not disclosed by the stack manual.
    membrane_thickness: float = 50e-6
    membrane_dry_density: float = 2000.0
    membrane_equivalent_weight: float = 1.10
    membrane_water_diffusivity: float = 2.0e-10
    # Regularization for the dry-end Springer conductivity singularity.
    # This assumption must be sensitivity-tested in V0.5.
    membrane_conductivity_floor: float = 5.0e-2
    anode_relative_humidity: float = 0.0
    membrane_current_density: float = 2.15e3

    @property
    def thickness_z(self) -> float:
        return self.gdl_thickness + self.cl_thickness

    @property
    def oxygen_inlet_concentration(self) -> float:
        return self.oxygen_mole_fraction * self.pressure / (
            self.gas_constant * self.stack_temperature
        )

    @property
    def d_o2_gdl(self) -> float:
        return self.d_o2_bulk * self.porosity_gdl**1.5

    @property
    def d_o2_cl(self) -> float:
        return self.d_o2_bulk * self.porosity_cl**1.5

    @property
    def beta_anodic(self) -> float:
        return self.alpha_anodic * self.faraday / (
            self.gas_constant * self.stack_temperature
        )

    @property
    def beta_cathodic(self) -> float:
        return self.alpha_cathodic * self.faraday / (
            self.gas_constant * self.stack_temperature
        )

    @property
    def target_stack_temperature_c(self) -> float:
        return self.tech.optimum_stack_temperature_c(self.stack_current_a)

    @property
    def target_air_flow_slpm(self) -> float:
        return self.stack.coolant_air_target_slpm(self.stack_current_a)

    @property
    def purge_period_s(self) -> float:
        return self.stack.purge_period_s(self.stack_current_a)

    @property
    def purge_volume_m3(self) -> float:
        return self.stack.purge_volume_m3()
