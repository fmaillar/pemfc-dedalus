"""Small, solver-independent PEMFC constitutive relations.

These functions are deliberately NumPy-only so they can be unit-tested without
starting Dedalus or MPI.
"""

from __future__ import annotations

import numpy as np

from .parameters import CathodeParameters


def cathode_overpotential(
    phi_s: np.ndarray | float,
    phi_m: np.ndarray | float,
    equilibrium_potential: float,
):
    """Return cathode activation overpotential eta = phi_s - phi_m - E_eq."""
    return np.asarray(phi_s) - np.asarray(phi_m) - equilibrium_potential


def butler_volmer_orr_current_density(
    c_o2: np.ndarray | float,
    phi_s: np.ndarray | float,
    phi_m: np.ndarray | float,
    params: CathodeParameters | None = None,
):
    """Return positive ORR volumetric current density [A/m^3].

    The sign convention is chosen so cathodic operation (eta < 0) gives a
    positive ORR magnitude.
    """
    p = params or CathodeParameters()
    c = np.asarray(c_o2)
    eta = cathode_overpotential(phi_s, phi_m, p.equilibrium_potential)
    activity = c / p.oxygen_inlet_concentration
    net = (
        p.j0_vol
        * activity**p.oxygen_reaction_order
        * (
            np.exp(-p.beta_cathodic * eta)
            - np.exp(p.beta_anodic * eta)
        )
    )
    eps = 1e-6 * p.j0_vol
    return 0.5 * (net + np.sqrt(net**2 + eps**2))


def oxygen_consumption_from_current(
    current_density: np.ndarray | float,
    faraday: float = CathodeParameters().faraday,
):
    """Convert ORR volumetric current density [A/m^3] to O2 sink [mol/m^3/s]."""
    return np.asarray(current_density) / (4.0 * faraday)
