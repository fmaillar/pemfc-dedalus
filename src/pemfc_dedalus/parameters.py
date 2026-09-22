"""Physical and geometrical parameters for the cathode validation case."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CathodeParameters:
    """Parameters for the first cathode GDL+CL diffusion-reaction model."""

    # Geometry [m]
    length_x: float = 2.0e-3
    length_y: float = 1.0e-3
    gdl_thickness: float = 200e-6
    cl_thickness: float = 12e-6

    # Gas / porous medium
    temperature: float = 353.15
    pressure: float = 1.5e5
    oxygen_mole_fraction: float = 0.21
    d_o2_bulk: float = 2.0e-5
    porosity_gdl: float = 0.70
    porosity_cl: float = 0.35

    # First validation sink: pseudo-first-order O2 consumption [1/s]
    # It is deliberately simple; Butler-Volmer replaces this in the next model.
    k_reaction: float = 250.0

    # Smooth transition width between GDL and CL [m].
    interface_width: float = 2.0e-6

    @property
    def thickness_z(self) -> float:
        return self.gdl_thickness + self.cl_thickness

    @property
    def oxygen_inlet_concentration(self) -> float:
        """Ideal-gas O2 concentration [mol/m^3]."""
        gas_constant = 8.31446261815324
        return self.oxygen_mole_fraction * self.pressure / (
            gas_constant * self.temperature
        )

    @property
    def d_o2_gdl(self) -> float:
        return self.d_o2_bulk * self.porosity_gdl**1.5

    @property
    def d_o2_cl(self) -> float:
        return self.d_o2_bulk * self.porosity_cl**1.5
