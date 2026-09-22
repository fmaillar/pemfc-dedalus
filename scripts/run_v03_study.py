#!/usr/bin/env python3
"""Run V0.3 membrane time- and spatial-convergence study."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pemfc_dedalus.convergence import (
    is_time_converged,
    relative_difference,
    scalar_last,
    scalar_relative_changes,
    time_convergence_metric,
)

V03_SCALARS = ("mean_lambda", "mean_sigma_m")


def run_command(args: list[str], *, env: dict[str, str]) -> int:
    """Run a child process while streaming output."""
    return subprocess.run(args, env=env, check=False).returncode


def run_and_validate(
    *,
    nz: int,
    stop_time: float,
    max_dt: float,
    scalar_dt: float,
    root: Path,
) -> dict[str, Any]:
    """Run one V0.3 membrane case and return its validation report."""
    output_dir = root / f"nz{nz}"
    report_path = output_dir / "report.json"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    rc = run_command(
        [
            "pemfc-membrane-1d",
            "--nz",
            str(nz),
            "--stop-time",
            str(stop_time),
            "--max-dt",
            str(max_dt),
            "--scalar-dt",
            str(scalar_dt),
            "--output-dir",
            str(output_dir),
        ],
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"V0.3 solver failed for nz={nz}")

    rc = run_command(
        [
            sys.executable,
            "scripts/validate_results.py",
            "--model",
            "v03",
            "--input",
            str(output_dir),
            "--output",
            str(report_path),
        ],
        env=env,
    )
    if not report_path.exists():
        raise RuntimeError(f"Validator did not produce {report_path}")

    report: dict[str, Any] = json.loads(report_path.read_text())
    if rc != 0 or not report.get("pass", False):
        raise RuntimeError(f"Physical validation failed for nz={nz}")
    return report


def converge_resolution(
    *,
    nz: int,
    initial_stop_time: float,
    max_stop_time: float,
    max_dt: float,
    scalar_dt: float,
    tolerance: float,
    root: Path,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Increase pseudo-time horizon until membrane scalars converge."""
    stop_time = initial_stop_time
    history: list[dict[str, Any]] = []
    while stop_time <= max_stop_time * (1.0 + 1e-12):
        report = run_and_validate(
            nz=nz,
            stop_time=stop_time,
            max_dt=max_dt,
            scalar_dt=scalar_dt,
            root=root,
        )
        changes = scalar_relative_changes(report, V03_SCALARS)
        metric = time_convergence_metric(report, V03_SCALARS)
        history.append(
            {
                "stop_time": stop_time,
                "metric": metric,
                "relative_last_step_change": changes,
            }
        )
        if is_time_converged(report, tolerance, V03_SCALARS):
            return report, stop_time, history
        stop_time *= 2.0

    raise RuntimeError(f"nz={nz} did not converge by t={max_stop_time:g} s")


def build_report(
    cases: list[dict[str, Any]],
    *,
    time_tolerance: float,
    grid_tolerance: float,
) -> dict[str, Any]:
    """Build compact V0.3 time/grid convergence summary."""
    finest_report = cases[-1]["validation"]
    for case in cases:
        report = case["validation"]
        case["relative_to_finest"] = {
            name: relative_difference(
                scalar_last(report, name),
                scalar_last(finest_report, name),
            )
            for name in V03_SCALARS
        }

    penultimate = cases[-2] if len(cases) >= 2 else cases[-1]
    grid_metric = max(penultimate["relative_to_finest"].values())
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "time_tolerance": time_tolerance,
        "grid_tolerance": grid_tolerance,
        "grid_metric_penultimate_to_finest": grid_metric,
        "grid_converged": grid_metric <= grid_tolerance,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nz",
        action="append",
        type=int,
        dest="resolutions",
        help="Chebyshev resolution; repeat for multiple resolutions",
    )
    parser.add_argument("--initial-stop-time", type=float, default=0.05)
    parser.add_argument("--max-stop-time", type=float, default=25.6)
    parser.add_argument("--max-dt", type=float, default=0.01)
    parser.add_argument("--scalar-dt", type=float, default=0.01)
    parser.add_argument("--time-tol", type=float, default=1e-5)
    parser.add_argument("--grid-tol", type=float, default=0.01)
    parser.add_argument("--work-dir", type=Path, default=Path(".study-output/v03"))
    parser.add_argument("--output", type=Path, default=Path("results/v03-study.json"))
    args = parser.parse_args()

    resolutions = args.resolutions or [32, 48, 64]
    if any(nz <= 0 for nz in resolutions):
        parser.error("all nz values must be positive")

    cases: list[dict[str, Any]] = []
    for nz in resolutions:
        report, stop_time, history = converge_resolution(
            nz=nz,
            initial_stop_time=args.initial_stop_time,
            max_stop_time=args.max_stop_time,
            max_dt=args.max_dt,
            scalar_dt=args.scalar_dt,
            tolerance=args.time_tol,
            root=args.work_dir,
        )
        cases.append(
            {
                "nz": nz,
                "converged_stop_time": stop_time,
                "time_convergence_history": history,
                "final_scalars": {
                    name: scalar_last(report, name) for name in V03_SCALARS
                },
                "validation": report,
            }
        )

    study = build_report(
        cases,
        time_tolerance=args.time_tol,
        grid_tolerance=args.grid_tol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(f"grid converged: {study['grid_converged']}")
    print(
        "penultimate->finest max relative difference: "
        f"{study['grid_metric_penultimate_to_finest']:.6g}"
    )
    raise SystemExit(0 if study["grid_converged"] else 2)


if __name__ == "__main__":
    main()
