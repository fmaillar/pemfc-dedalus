"""Geometry-derived cathode air heat-transfer closure for V11.

FCgen-1020ACS cathode-channel dimensions are taken from the published
Andisheh-Tadbir geometry. Heat transfer uses the Sadasivam fully-developed
laminar trapezoidal-duct Nusselt correlation. No fitted hA or prescribed NTU
is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .v11_thermal import standard_air_mass_flow_kg_s


@dataclass(frozen=True)
class V11CathodeChannelGeometry:
    """Published FCgen-1020ACS cathode-channel geometry."""

    height_m: float = 2.5e-3
    wide_base_m: float = 2.5e-3
    narrow_base_m: float = 1.5e-3
    length_m: float = 60.0e-3
    channels_per_cell: int = 80

    @property
    def cross_section_m2(self) -> float:
        return 0.5 * (self.wide_base_m + self.narrow_base_m) * self.height_m

    @property
    def sloping_side_m(self) -> float:
        half_base_difference = 0.5 * (
            self.wide_base_m - self.narrow_base_m
        )
        return math.hypot(self.height_m, half_base_difference)

    @property
    def wetted_perimeter_m(self) -> float:
        return (
            self.wide_base_m
            + self.narrow_base_m
            + 2.0 * self.sloping_side_m
        )

    @property
    def hydraulic_diameter_m(self) -> float:
        return 4.0 * self.cross_section_m2 / self.wetted_perimeter_m

    @property
    def aspect_ratio(self) -> float:
        return self.height_m / self.wide_base_m

    @property
    def convective_area_per_cell_m2(self) -> float:
        return (
            self.wetted_perimeter_m
            * self.length_m
            * self.channels_per_cell
        )


@dataclass(frozen=True)
class V11CathodeHeatTransfer:
    """Resolved geometry-based cathode heat-transfer state."""

    nusselt: float
    hydraulic_diameter_m: float
    convective_area_per_cell_m2: float
    heat_transfer_coefficient_w_m2_k: float
    ua_stack_w_k: float
    air_capacity_rate_w_k: float
    ntu: float
    effectiveness: float
    outlet_temperature_k: float


def air_thermal_conductivity_w_m_k(temperature_k: float) -> float:
    """Return a linear air-conductivity approximation over PEMFC temperatures.

    Literature values 0.0264 W/(m K) at 300 K and 0.0335 W/(m K) at 400 K
    are used directly; this relation is not calibrated to stack data.
    """
    if not 250.0 <= temperature_k <= 400.0:
        raise ValueError("temperature_k must lie in [250, 400] K")
    return 0.0264 + (temperature_k - 300.0) * (0.0335 - 0.0264) / 100.0


def trapezoidal_fully_developed_nusselt(aspect_ratio: float) -> float:
    """Return Sadasivam isothermal-wall Nusselt number for a trapezoidal duct."""
    if aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive")
    gamma = aspect_ratio
    value = (
        7.541
        - 20.183 * gamma
        + 33.483 * gamma**2
        - 26.286 * gamma**3
        + 9.9036 * gamma**4
        - 1.4825 * gamma**5
    )
    if value <= 0.0:
        raise ValueError("Nusselt correlation returned a non-positive value")
    return value


def cathode_air_outlet_temperature_geometry_v11(
    *,
    stack_temperature_k: float,
    inlet_temperature_k: float,
    stack_air_flow_slpm: float,
    n_cells: int,
    geometry: V11CathodeChannelGeometry | None = None,
    air_specific_heat_j_kg_k: float = 1005.0,
) -> V11CathodeHeatTransfer:
    """Predict cathode air outlet temperature from channel geometry.

    For an isothermal wall and constant properties,

        T_out = T_stack - (T_stack - T_in) exp(-NTU),

    where NTU = UA/(m_dot cp), with U obtained from Nu*k/Dh.
    """
    if stack_temperature_k <= 0.0 or inlet_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    if stack_temperature_k < inlet_temperature_k:
        raise ValueError(
            "geometry closure currently requires stack temperature >= inlet"
        )
    if stack_air_flow_slpm <= 0.0:
        raise ValueError("stack_air_flow_slpm must be positive")
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if air_specific_heat_j_kg_k <= 0.0:
        raise ValueError("air_specific_heat_j_kg_k must be positive")

    geom = V11CathodeChannelGeometry() if geometry is None else geometry
    nusselt = trapezoidal_fully_developed_nusselt(geom.aspect_ratio)
    film_temperature = 0.5 * (stack_temperature_k + inlet_temperature_k)
    conductivity = air_thermal_conductivity_w_m_k(film_temperature)
    h_conv = nusselt * conductivity / geom.hydraulic_diameter_m
    ua_stack = (
        h_conv * geom.convective_area_per_cell_m2 * n_cells
    )
    mass_flow = standard_air_mass_flow_kg_s(
        stack_air_flow_slpm=stack_air_flow_slpm
    )
    capacity_rate = mass_flow * air_specific_heat_j_kg_k
    ntu = ua_stack / capacity_rate
    effectiveness = 1.0 - math.exp(-ntu)
    outlet = inlet_temperature_k + effectiveness * (
        stack_temperature_k - inlet_temperature_k
    )

    return V11CathodeHeatTransfer(
        nusselt=nusselt,
        hydraulic_diameter_m=geom.hydraulic_diameter_m,
        convective_area_per_cell_m2=geom.convective_area_per_cell_m2,
        heat_transfer_coefficient_w_m2_k=h_conv,
        ua_stack_w_k=ua_stack,
        air_capacity_rate_w_k=capacity_rate,
        ntu=ntu,
        effectiveness=effectiveness,
        outlet_temperature_k=outlet,
    )
