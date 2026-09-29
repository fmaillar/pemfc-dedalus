"""Hydrogen-dilution feedback closures for the reduced V0.7 anode model."""

from __future__ import annotations


def hydrogen_partial_pressure_feedback_factor(
    actual_partial_pressure_pa: float,
    reference_partial_pressure_pa: float,
    exponent: float,
) -> float:
    """Return a normalized power-law current factor.

    The closure is

        f_H2 = (p_H2 / p_H2,ref) ** exponent.

    An exponent of zero exactly disables the feedback. The reference pressure
    is evaluated for the same water state but with no N2, so the factor isolates
    H2 dilution by inert gas rather than double-counting the RH dependence
    already present in the V0.6 closure.

    This is a screening law, not a calibrated HOR kinetic model.
    """
    if actual_partial_pressure_pa < 0.0:
        raise ValueError("actual_partial_pressure_pa must be non-negative")
    if reference_partial_pressure_pa <= 0.0:
        raise ValueError("reference_partial_pressure_pa must be positive")
    if exponent < 0.0:
        raise ValueError("exponent must be non-negative")
    if exponent == 0.0:
        return 1.0
    ratio = max(actual_partial_pressure_pa / reference_partial_pressure_pa, 0.0)
    return ratio**exponent
