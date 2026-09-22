#!/usr/bin/env python3
"""Validate Dedalus HDF5 outputs and emit a compact, git-friendly JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

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
    }
    if finite.any():
        f = flat[finite]
        result.update(
            {
                "min": float(np.min(f)),
                "max": float(np.max(f)),
                "mean": float(np.mean(f)),
                "std": float(np.std(f)),
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


def read_scalar_series(path: Path) -> dict:
    out = {}
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


def validate(model: str, root: Path) -> dict:
    p = CathodeParameters()
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
        add_check(
            checks,
            "oxygen concentration non-negative",
            c["min"] >= -1e-8 * p.oxygen_inlet_concentration,
            c["min"],
            ">= approximately 0",
        )
        add_check(
            checks,
            "oxygen concentration not above open-air boundary by >5%",
            c["max"] <= 1.05 * p.oxygen_inlet_concentration,
            c["max"],
            1.05 * p.oxygen_inlet_concentration,
        )

    if model == "v02":
        required = ("phi_s", "phi_m", "eta", "j_orr")
        for name in required:
            add_check(checks, f"required field {name} present", name in fields)

        j = fields.get("j_orr")
        if j:
            scale = max(abs(j.get("max", 0.0)), 1.0)
            add_check(
                checks,
                "ORR current is non-negative",
                j["min"] >= -1e-10 * scale,
                j["min"],
                ">= 0 within numerical tolerance",
            )
            add_check(
                checks,
                "ORR current is active",
                j["max"] > 0.0,
                j["max"],
                "> 0",
            )

        eta = fields.get("eta")
        if eta:
            add_check(
                checks,
                "mean cathode overpotential is cathodic",
                eta["mean"] < 0.0,
                eta["mean"],
                "< 0 V",
            )

        phi_s = fields.get("phi_s")
        phi_m = fields.get("phi_m")
        if phi_s:
            add_check(
                checks,
                "solid potential remains in broad physical range",
                -0.5 <= phi_s["min"] and phi_s["max"] <= 2.0,
                [phi_s["min"], phi_s["max"]],
                "[-0.5, 2.0] V",
            )
        if phi_m:
            add_check(
                checks,
                "protonic potential remains in broad physical range",
                -1.0 <= phi_m["min"] and phi_m["max"] <= 1.0,
                [phi_m["min"], phi_m["max"]],
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

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
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
    parser.add_argument("--model", choices=("v01", "v02"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate(args.model, args.input)
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
