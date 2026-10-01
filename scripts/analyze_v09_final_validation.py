"""Validate the final V0.9 quasi-3D factorial baseline."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

INPUT = Path("results/quick-v09-quasi3d-full-map.json")
OUTPUT = Path("results/v09-final-validation.json")


def _case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[float, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (
            float(row["current_a"]),
            float(row["inlet_temperature_c"]),
            float(row["ntu"]),
        )
        unique.setdefault(key, row)
    return list(unique.values())


def _assert_monotonic(
    cases: list[dict[str, Any]],
    *,
    group_keys: tuple[str, str],
    sort_key: str,
    increasing: bool,
) -> None:
    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        group = (
            float(row[group_keys[0]]),
            float(row[group_keys[1]]),
        )
        groups[group].append(row)

    for group, values in groups.items():
        ordered = sorted(values, key=lambda row: float(row[sort_key]))
        spans = [float(row["reaction_current_span"]) for row in ordered]
        pairs = zip(spans, spans[1:], strict=False)
        if increasing:
            assert all(right >= left for left, right in pairs), group
        else:
            assert all(right <= left for left, right in pairs), group


def main() -> None:
    data = json.loads(INPUT.read_text())
    rows = data["rows"]
    skipped = data["skipped"]
    cases = _case_rows(rows)

    assert data["case_count"] == 24
    assert len(cases) == 24
    assert len(skipped) == 3
    assert {
        (
            float(item["current_a"]),
            float(item["inlet_temperature_c"]),
            float(item["ntu"]),
        )
        for item in skipped
    } == {
        (7.3, 30.0, 1.0),
        (7.3, 30.0, 3.0),
        (7.3, 30.0, 5.0),
    }

    _assert_monotonic(
        cases,
        group_keys=("inlet_temperature_c", "ntu"),
        sort_key="current_a",
        increasing=True,
    )
    _assert_monotonic(
        cases,
        group_keys=("current_a", "ntu"),
        sort_key="inlet_temperature_c",
        increasing=False,
    )
    _assert_monotonic(
        cases,
        group_keys=("current_a", "inlet_temperature_c"),
        sort_key="ntu",
        increasing=True,
    )

    span_min = min(float(row["reaction_current_span"]) for row in cases)
    span_max = max(float(row["reaction_current_span"]) for row in cases)

    assert abs(span_min - float(data["span_min"])) < 1.0e-12
    assert abs(span_max - float(data["span_max"])) < 1.0e-12
    assert 0.009 < span_min < 0.010
    assert 0.106 < span_max < 0.108

    summary = {
        "schema_version": 1,
        "model": "v09-final-validation",
        "status": "pass",
        "reachable_case_count": len(cases),
        "skipped_case_count": len(skipped),
        "span_min": span_min,
        "span_max": span_max,
        "monotonic_with_current": True,
        "monotonic_with_inlet_temperature": True,
        "monotonic_with_ntu": True,
        "freeze_ready": True,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "V0.9 final validation passed: "
        f"{len(cases)} reachable cases, "
        f"span={100.0 * span_min:.3f}%..{100.0 * span_max:.3f}%",
        flush=True,
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
