#!/usr/bin/env python3
"""Run V0.2 to pseudo-time convergence on successively refined grids."""

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
    DEFAULT_SCALARS,
    is_time_converged,
    relative_difference,
    scalar_last,
    scalar_relative_changes,
    time_convergence_metric,
)


def parse_grid(value: str) -> tuple[int, int, int]:
    """Parse NXxNYxNZ grid syntax."""
    try:
        nx, ny, nz = (int(item) for item in value.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("grid must have form NXxNYxNZ") from exc
    if min(nx, ny, nz) <= 0:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return nx, ny, nz


def run_command(args: list[str], *, env: dict[str, str]) -> int:
    """Run a child process while streaming its output."""
    return subprocess.run(args, env=env, check=False).returncode


def mpi_ranks_for_grid(grid: tuple[int, int, int], max_ranks: int) -> int:
    """Choose a practical MPI rank count for the grid, capped by max_ranks."""
    nx, ny, _ = grid
    transverse_modes = nx * ny
    if transverse_modes <= 64:
        desired = 2
    elif transverse_modes <= 144:
        desired = 4
    else:
        desired = 8
    return min(desired, max_ranks)


def run_and_validate(
    *,
    grid: tuple[int, int, int],
    stop_time: float,
    max_dt: float,
    scalar_dt: float,
    root: Path,
    mpiexec: str,
    mpi_flags: list[str],
    mpi_n: int,
) -> dict[str, Any]:
    """Run one V0.2 case and return its validation report."""
    nx, ny, nz = grid
    output_dir = root / f"{nx}x{ny}x{nz}"
    report_path = output_dir / "report.json"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    ranks = mpi_ranks_for_grid(grid, mpi_n)
    print(
        f"grid {nx}x{ny}x{nz}: using {ranks} MPI ranks "
        f"(configured maximum {mpi_n})"
    )

    rc = run_command(
        [
            mpiexec,
            *mpi_flags,
            "-n",
            str(ranks),
            "pemfc-cathode-electrochem-3d",
            "--nx",
            str(nx),
            "--ny",
            str(ny),
            "--nz",
            str(nz),
            "--stop-time",
            str(stop_time),
            "--max-dt",
            str(max_dt),
            "--output-dir",
            str(output_dir),
            "--scalar-dt",
            str(scalar_dt),
        ],
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"V0.2 solver failed for grid {nx}x{ny}x{nz}")

    rc = run_command(
        [
            sys.executable,
            "scripts/validate_results.py",
            "--model",
            "v02",
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
        raise RuntimeError(f"Physical validation failed for grid {nx}x{ny}x{nz}")
    return report


def converge_grid(
    *,
    grid: tuple[int, int, int],
    initial_stop_time: float,
    max_stop_time: float,
    max_dt: float,
    tolerance: float,
    scalar_dt: float,
    root: Path,
    mpiexec: str,
    mpi_flags: list[str],
    mpi_n: int,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Repeat a run with increasing horizon until scalar changes converge."""
    stop_time = initial_stop_time
    history: list[dict[str, Any]] = []
    while stop_time <= max_stop_time * (1.0 + 1e-12):
        report = run_and_validate(
            grid=grid,
            stop_time=stop_time,
            max_dt=max_dt,
            scalar_dt=scalar_dt,
            root=root,
            mpiexec=mpiexec,
            mpi_flags=mpi_flags,
            mpi_n=mpi_n,
        )
        changes = scalar_relative_changes(report)
        metric = time_convergence_metric(report)
        history.append(
            {
                "stop_time": stop_time,
                "metric": metric,
                "relative_last_step_change": changes,
            }
        )
        if is_time_converged(report, tolerance):
            return report, stop_time, history
        stop_time *= 2.0
    raise RuntimeError(
        f"Grid {grid[0]}x{grid[1]}x{grid[2]} did not converge by t={max_stop_time:g}"
    )


def build_study_report(
    cases: list[dict[str, Any]],
    *,
    time_tolerance: float,
    grid_tolerance: float,
) -> dict[str, Any]:
    """Build a compact time- and grid-convergence summary."""
    finest = cases[-1]
    finest_report = finest["validation"]
    for case in cases:
        report = case["validation"]
        case["relative_to_finest"] = {
            name: relative_difference(
                scalar_last(report, name),
                scalar_last(finest_report, name),
            )
            for name in DEFAULT_SCALARS
        }

    coarser = cases[-2] if len(cases) >= 2 else cases[-1]
    grid_metric = max(coarser["relative_to_finest"].values())
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
        "--grid",
        action="append",
        type=parse_grid,
        dest="grids",
        help="grid NXxNYxNZ; repeat for multiple resolutions",
    )
    parser.add_argument("--initial-stop-time", type=float, default=5e-5)
    parser.add_argument("--max-stop-time", type=float, default=0.0128)
    parser.add_argument("--max-dt", type=float, default=1e-6)
    parser.add_argument("--time-tol", type=float, default=1e-5)
    parser.add_argument("--grid-tol", type=float, default=0.02)
    parser.add_argument(
        "--scalar-dt",
        type=float,
        default=1e-5,
        help="fixed scalar sampling interval used by the convergence criterion",
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".study-output/v02"))
    parser.add_argument("--output", type=Path, default=Path("results/v02-study.json"))
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument(
        "--mpi-flags",
        default=os.environ.get("MPI_FLAGS", "--use-hwthread-cpus"),
        help="space-separated flags passed to mpiexec",
    )
    parser.add_argument(
        "--mpi-n",
        type=int,
        default=int(os.environ.get("MPI_N", "8")),
        help="maximum number of MPI ranks for each Dedalus run",
    )
    args = parser.parse_args()

    grids = args.grids or [(8, 8, 24), (12, 12, 36), (16, 16, 48)]
    if args.mpi_n <= 0:
        parser.error("--mpi-n must be positive")
    mpi_flags = args.mpi_flags.split()
    cases: list[dict[str, Any]] = []
    for grid in grids:
        report, stop_time, history = converge_grid(
            grid=grid,
            initial_stop_time=args.initial_stop_time,
            max_stop_time=args.max_stop_time,
            max_dt=args.max_dt,
            tolerance=args.time_tol,
            scalar_dt=args.scalar_dt,
            root=args.work_dir,
            mpiexec=args.mpiexec,
            mpi_flags=mpi_flags,
            mpi_n=args.mpi_n,
        )
        cases.append(
            {
                "grid": list(grid),
                "mpi_ranks": mpi_ranks_for_grid(grid, args.mpi_n),
                "converged_stop_time": stop_time,
                "time_convergence_history": history,
                "final_scalars": {
                    name: scalar_last(report, name) for name in DEFAULT_SCALARS
                },
                "validation": report,
            }
        )

    study = build_study_report(
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
