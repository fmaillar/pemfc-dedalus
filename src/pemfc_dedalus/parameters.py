"""Physical and geometrical parameters for PEMFC validation models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CathodeParameters:
    """Parameters for the cathode GDL+CL models."""

    # Geometry [m]
    length_x: float = 2.0e-3
    length_y: float = 1.0e-3
    gdl_thickness: float = 200e-6
    cl_thickness: float = 12e-6

    # Open-cathode gas conditions
    temperature: float = 353.15
    pressure: float = 1.5e5
    oxygen_mole_fraction: float = 0.21
    d_o2_bulk: float = 2.0e-5
    porosity_gdl: float = 0.70
    porosity_cl: float = 0.35

    # V0.1 pseudo-first-order sink [1/s]
    k_reaction: float = 250.0

    # Electrochemistry, V0.2
    faraday: float = 96485.33212
    gas_constant: float = 8.31446261815324
    equilibrium_potential: float = 1.18
    cathode_solid_potential: float = 0.78
    membrane_proton_potential: float = 0.0
    alpha_anodic: float = 0.5
    alpha_cathodic: float = 0.5

    # Volumetric exchange-current scale [A/m^3].
    # This is a validation-scale effective parameter, not yet a calibrated CL law.
    j0_vol: float = 1.0e5
    oxygen_reaction_order: float = 1.0

    # Effective conductivities [S/m]
    sigma_s_gdl: float = 1.0e3
    sigma_s_cl: float = 3.0e2
    sigma_m_cl: float = 5.0
    # Numerical regularisation outside the CL. Proton transport there is not physical.
    sigma_m_floor: float = 1.0e-2

    # Pseudo-capacitances [F/m^3] used only for pseudo-transient convergence.
    pseudo_capacitance_s: float = 1.0e7
    pseudo_capacitance_m: float = 1.0e7

    # Smooth transition width between GDL and CL [m]
    interface_width: float = 2.0e-6

    @property
    def thickness_z(self) -> float:
        return self.gdl_thickness + self.cl_thickness

    @property
    def oxygen_inlet_concentration(self) -> float:
        """Ideal-gas O2 concentration [mol/m^3]."""
        return self.oxygen_mole_fraction * self.pressure / (
            self.gas_constant * self.temperature
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
            self.gas_constant * self.temperature
        )

    @property
    def beta_cathodic(self) -> float:
        return self.alpha_cathodic * self.faraday / (
            self.gas_constant * self.temperature
        )
