"""Convergence helpers for PEMFC pseudo-transient and grid studies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_SCALARS = ("mean_c_o2", "mean_eta", "total_reaction_current")


def scalar_relative_changes(
    report: Mapping[str, Any],
    names: tuple[str, ...] = DEFAULT_SCALARS,
) -> dict[str, float]:
    """Return last-step relative changes for selected scalar observables."""
    scalars = report.get("scalar_series", {})
    changes: dict[str, float] = {}
    for name in names:
        stats = scalars.get(name)
        if not isinstance(stats, Mapping):
            raise KeyError(f"Missing scalar series: {name}")
        value = stats.get("relative_last_step_change")
        if value is None:
            raise KeyError(f"Missing relative_last_step_change for scalar: {name}")
        changes[name] = float(value)
    return changes


def time_convergence_metric(
    report: Mapping[str, Any],
    names: tuple[str, ...] = DEFAULT_SCALARS,
) -> float:
    """Return the maximum selected scalar last-step relative change."""
    return max(scalar_relative_changes(report, names).values())


def is_time_converged(
    report: Mapping[str, Any],
    tolerance: float,
    names: tuple[str, ...] = DEFAULT_SCALARS,
) -> bool:
    """Return whether all selected scalar changes are below tolerance."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    return time_convergence_metric(report, names) <= tolerance


def relative_difference(value: float, reference: float) -> float:
    """Return a scale-safe relative difference against a reference value."""
    scale = max(abs(reference), 1e-30)
    return abs(value - reference) / scale


def scalar_last(report: Mapping[str, Any], name: str) -> float:
    """Extract the final value of one scalar series from a validation report."""
    scalars = report.get("scalar_series", {})
    stats = scalars.get(name)
    if not isinstance(stats, Mapping) or "last" not in stats:
        raise KeyError(f"Missing final scalar value: {name}")
    return float(stats["last"])
