#!/usr/bin/env python3
"""Validate Dedalus HDF5 outputs and emit a compact, git-friendly JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pemfc_dedalus.membrane import membrane_water_content_from_activity
from pemfc_dedalus.parameters import CathodeParameters


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def latest_h5(directory: Path) -> Path:
    files = sorted(directory.glob("*_s*.h5"))
    if not files:
        raise FileNotFoundError(f"No Dedalus HDF5 file found in {directory}")
    return files[-1]


def finite_stats(data) -> dict:
    a = np.asarray(data)
    if np.iscomplexobj(a):
        a = np.real_if_close(a)
    a = np.asarray(a, dtype=float)
    flat = a.ravel()
    finite = np.isfinite(flat)
    result = {
        "size": int(flat.size),
        "finite_fraction": float(finite.mean()) if flat.size else 0.0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "p01": None,
        "p50": None,
        "p99": None,
    }
    if finite.any():
        f = flat[finite]
        result.update(
            {
                "min": float(np.min(f)),
                "max": float(np.max(f)),
                "mean": float(np.mean(f)),
                "std": float(np.std(f, dtype=np.float64)) if np.max(np.abs(f)) < 1e150 else None,
                "p01": float(np.percentile(f, 1)),
                "p50": float(np.percentile(f, 50)),
                "p99": float(np.percentile(f, 99)),
            }
        )
    return result


def read_last_tasks(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as h5:
        for name, ds in h5["tasks"].items():
            out[name] = finite_stats(ds[-1])
    return out


def read_scalar_series(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as h5:
        times = np.asarray(h5["scales/sim_time"])
        out["sim_time"] = finite_stats(times)
        if times.size:
            out["final_sim_time"] = float(times[-1])
        for name, ds in h5["tasks"].items():
            a = np.asarray(ds).reshape(len(ds), -1)
            series = np.mean(np.real_if_close(a), axis=1).astype(float)
            stats = finite_stats(series)
            if series.size:
                stats["first"] = float(series[0])
                stats["last"] = float(series[-1])
                if series.size >= 2:
                    denom = max(abs(float(series[-1])), 1e-30)
                    stats["relative_last_step_change"] = float(
                        abs(series[-1] - series[-2]) / denom
                    )
            out[name] = stats
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def add_check(checks: list[dict], name: str, passed: bool, value=None, limit=None):
    checks.append(
        {
            "name": name,
            "pass": bool(passed),
            "value": value,
            "limit": limit,
        }
    )


def validate(
    model: str,
    root: Path,
    *,
    relative_humidity: float | None = None,
) -> dict:
    p = CathodeParameters()
    if relative_humidity is not None:
        if not 0.0 <= relative_humidity <= 1.0:
            raise ValueError("relative_humidity must be between 0 and 1")
        p = CathodeParameters(relative_humidity=relative_humidity)
    snap = latest_h5(root / "snapshots")
    scalar = latest_h5(root / "scalars")

    fields = read_last_tasks(snap)
    scalars = read_scalar_series(scalar)
    checks: list[dict] = []

    for field_name, stats in fields.items():
        add_check(
            checks,
            f"{field_name}: all values finite",
            stats["finite_fraction"] == 1.0,
            stats["finite_fraction"],
            1.0,
        )

    c = fields.get("c_o2")
    if c:
        c_min = c.get("min")
        c_max = c.get("max")
        add_check(
            checks,
            "oxygen concentration has finite extrema",
            c_min is not None and c_max is not None,
            [c_min, c_max],
            "finite",
        )
        if c_min is not None and c_max is not None:
            add_check(
                checks,
                "oxygen concentration non-negative",
                c_min >= -1e-8 * p.oxygen_inlet_concentration,
                c_min,
                ">= approximately 0",
            )
            add_check(
                checks,
                "oxygen concentration not above open-air boundary by >5%",
                c_max <= 1.05 * p.oxygen_inlet_concentration,
                c_max,
                1.05 * p.oxygen_inlet_concentration,
            )

    if model in ("v02", "v04"):
        required_electrochem = ("phi_s", "phi_m", "eta", "j_orr")
        for name in required_electrochem:
            add_check(checks, f"required field {name} present", name in fields)

        j = fields.get("j_orr")
        if j:
            j_min = j.get("min")
            j_max = j.get("max")
            add_check(
                checks,
                "ORR current has finite extrema",
                j_min is not None and j_max is not None,
                [j_min, j_max],
                "finite",
            )
            if j_min is not None and j_max is not None:
                add_check(
                    checks,
                    "ORR current is active",
                    j_max > 0.0,
                    j_max,
                    "> 0",
                )
                # Nonlinear products are projected back onto the truncated
                # spectral basis, so a strictly positive pointwise source can
                # show a small Gibbs/aliasing undershoot.  Treat this as a
                # spectral-resolution diagnostic rather than requiring an
                # impossible exact positivity invariant after projection.
                negative_fraction = max(0.0, -j_min) / max(abs(j_max), 1.0)
                add_check(
                    checks,
                    "ORR negative spectral undershoot is limited",
                    negative_fraction <= 0.05,
                    negative_fraction,
                    "<= 0.05 of positive peak",
                )

        eta = fields.get("eta")
        if eta:
            eta_mean = eta.get("mean")
            add_check(
                checks,
                "mean cathode overpotential is finite",
                eta_mean is not None,
                eta_mean,
                "finite",
            )
            if eta_mean is not None:
                add_check(
                    checks,
                    "mean cathode overpotential is cathodic",
                    eta_mean < 0.0,
                    eta_mean,
                    "< 0 V",
                )

        phi_s = fields.get("phi_s")
        phi_m = fields.get("phi_m")
        if phi_s:
            ps_min, ps_max = phi_s.get("min"), phi_s.get("max")
            add_check(
                checks,
                "solid potential has finite extrema",
                ps_min is not None and ps_max is not None,
                [ps_min, ps_max],
                "finite",
            )
            if ps_min is not None and ps_max is not None:
                add_check(
                    checks,
                    "solid potential remains in broad physical range",
                    -0.5 <= ps_min and ps_max <= 2.0,
                    [ps_min, ps_max],
                    "[-0.5, 2.0] V",
                )
        if phi_m:
            pm_min, pm_max = phi_m.get("min"), phi_m.get("max")
            add_check(
                checks,
                "protonic potential has finite extrema",
                pm_min is not None and pm_max is not None,
                [pm_min, pm_max],
                "finite",
            )
            if pm_min is not None and pm_max is not None:
                add_check(
                    checks,
                    "protonic potential remains in broad physical range",
                    -1.0 <= pm_min and pm_max <= 1.0,
                    [pm_min, pm_max],
                    "[-1.0, 1.0] V",
                )

        total = scalars.get("total_reaction_current")
        if total and "last" in total:
            add_check(
                checks,
                "integrated reaction current is positive",
                total["last"] > 0.0,
                total["last"],
                "> 0 A (domain integral)",
            )

    if model == "v04":
        required_v04 = ("lambda_cl", "sigma_m")
        for name in required_v04:
            add_check(checks, f"required field {name} present", name in fields)

        lambda_cl = fields.get("lambda_cl")
        if lambda_cl:
            lam_min = lambda_cl.get("min")
            lam_max = lambda_cl.get("max")
            add_check(
                checks,
                "hydrated cathode lambda has finite extrema",
                lam_min is not None and lam_max is not None,
                [lam_min, lam_max],
                "finite",
            )
            if lam_min is not None and lam_max is not None:
                target_lambda = membrane_water_content_from_activity(
                    p.relative_humidity
                ).item()
                add_check(
                    checks,
                    "hydrated cathode lambda remains bounded",
                    lam_min >= -1e-8 and lam_max <= 1.01 * target_lambda,
                    [lam_min, lam_max],
                    [0.0, 1.01 * target_lambda],
                )

        sigma = fields.get("sigma_m")
        if sigma:
            sigma_min = sigma.get("min")
            sigma_max = sigma.get("max")
            add_check(
                checks,
                "hydrated cathode conductivity is positive",
                sigma_min is not None and sigma_max is not None
                and sigma_min > 0.0 and sigma_max > sigma_min,
                [sigma_min, sigma_max],
                "> 0 S/m with CL/GDL contrast",
            )

    if model == "v03":
        required_v03 = ("lambda", "sigma_m", "n_drag")
        for name in required_v03:
            add_check(checks, f"required field {name} present", name in fields)

        lam = fields.get("lambda")
        lambda_anode = membrane_water_content_from_activity(
            p.anode_relative_humidity
        ).item()
        lambda_cathode = membrane_water_content_from_activity(
            p.relative_humidity
        ).item()
        if lam:
            lam_min, lam_max = lam.get("min"), lam.get("max")
            add_check(
                checks,
                "membrane water content has finite extrema",
                lam_min is not None and lam_max is not None,
                [lam_min, lam_max],
                "finite",
            )
            if lam_min is not None and lam_max is not None:
                span = max(lambda_cathode - lambda_anode, 1.0)
                tol = 1e-3 * span
                add_check(
                    checks,
                    "membrane water content remains within boundary range",
                    lam_min >= lambda_anode - tol and lam_max <= lambda_cathode + tol,
                    [lam_min, lam_max],
                    [lambda_anode - tol, lambda_cathode + tol],
                )

        sigma = fields.get("sigma_m")
        if sigma:
            sigma_min = sigma.get("min")
            sigma_max = sigma.get("max")
            add_check(
                checks,
                "membrane proton conductivity has finite extrema",
                sigma_min is not None and sigma_max is not None,
                [sigma_min, sigma_max],
                "finite",
            )
            if sigma_min is not None and sigma_max is not None:
                negative_fraction = max(0.0, -sigma_min) / max(abs(sigma_max), 1e-30)
                add_check(
                    checks,
                    "membrane conductivity negative spectral undershoot is limited",
                    negative_fraction <= 0.01,
                    negative_fraction,
                    "<= 0.01 of positive peak",
                )

        drag = fields.get("n_drag")
        if drag and drag.get("min") is not None:
            add_check(
                checks,
                "electro-osmotic drag coefficient is non-negative",
                drag["min"] >= -1e-10,
                drag["min"],
                ">= approximately 0",
            )

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "git_commit": git_commit(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "input": {
            "root": str(root),
            "snapshots": str(snap),
            "scalars": str(scalar),
            "snapshots_sha256": sha256(snap),
            "scalars_sha256": sha256(scalar),
        },
        "fields_last_write": fields,
        "scalar_series": scalars,
        "checks": checks,
        "pass": all(item["pass"] for item in checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("v01", "v02", "v03", "v04"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--relative-humidity",
        type=float,
        default=None,
        help="RH override used to validate V0.4 hydration-dependent bounds",
    )
    args = parser.parse_args()

    report = validate(
        args.model,
        args.input,
        relative_humidity=args.relative_humidity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(f"overall pass: {report['pass']}")
    failed = [c for c in report["checks"] if not c["pass"]]
    for check in failed:
        print(f"FAIL: {check['name']}: value={check['value']} limit={check['limit']}")
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
