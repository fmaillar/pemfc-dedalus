"""Generate publication-ready figures for the V0.4 RH sweep analysis."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_polarization(rows: list[dict[str, float]], output_dir: Path) -> None:
    by_rh: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_rh[row["rh_percent"]].append(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for rh in sorted(by_rh):
        points = sorted(by_rh[rh], key=lambda row: row["cathode_solid_potential_v"])
        ax.plot(
            [row["current_density_a_per_cm2"] for row in points],
            [row["cathode_solid_potential_v"] for row in points],
            marker="o",
            label=f"RH {rh:.0f}%",
        )

    ax.set_xlabel("Current density (A cm$^{-2}$)")
    ax.set_ylabel("Cathode solid potential (V)")
    ax.set_title("V0.4 polarization curves by relative humidity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "polarization")


def plot_rh_sensitivity(rows: list[dict[str, float]], output_dir: Path) -> None:
    by_voltage: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_voltage[row["cathode_solid_potential_v"]].append(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for voltage in sorted(by_voltage):
        points = sorted(by_voltage[voltage], key=lambda row: row["rh_percent"])
        ax.plot(
            [row["rh_percent"] for row in points],
            [row["dI_dRH_ma_per_percent"] for row in points],
            marker="o",
            label=f"{voltage:.3f} V",
        )

    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel(r"$\partial I/\partial RH$ (mA per %-point)")
    ax.set_title("Current sensitivity to cathode relative humidity")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Cathode potential")
    save_figure(fig, output_dir / "rh_sensitivity")


def plot_sigma_elasticity(rows: list[dict[str, float]], output_dir: Path) -> None:
    by_voltage: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_voltage[row["cathode_solid_potential_v"]].append(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for voltage in sorted(by_voltage):
        points = sorted(by_voltage[voltage], key=lambda row: row["rh_percent"])
        ax.plot(
            [row["rh_percent"] for row in points],
            [row["dlnI_dlnsigma"] for row in points],
            marker="o",
            label=f"{voltage:.3f} V",
        )

    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel(r"$\partial\ln I/\partial\ln \sigma_m$")
    ax.set_title("Elasticity of current to protonic conductivity")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Cathode potential")
    save_figure(fig, output_dir / "sigma_elasticity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/analysis-v04-rh"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures-v04-rh"),
    )
    args = parser.parse_args()

    polarization = read_rows(args.analysis_dir / "polarization.csv")
    sensitivity = read_rows(args.analysis_dir / "sensitivity.csv")

    plot_polarization(polarization, args.output_dir)
    plot_rh_sensitivity(sensitivity, args.output_dir)
    plot_sigma_elasticity(sensitivity, args.output_dir)

    for stem in ("polarization", "rh_sensitivity", "sigma_elasticity"):
        print(f"Wrote {args.output_dir / (stem + '.png')}")
        print(f"Wrote {args.output_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
