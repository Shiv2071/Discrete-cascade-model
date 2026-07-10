"""Generate white-paper and dark-website figures from audited DSCD artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "model_simulation" / "dscd_cosmology_results"
FIGURES = ROOT / "figures"
DARK_FIGURES = FIGURES / "dark fig"


def load(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


PILOT = load("pilot_ensemble.json")
RETROSPECTIVE = load("retrospective_quick.json")
AUDIT = load("audit.json")


def apply_style(dark: bool) -> None:
    background = "#101114" if dark else "#ffffff"
    foreground = "#e7e8ea" if dark else "#17181b"
    grid = "#34363d" if dark else "#d9dce2"
    plt.rcParams.update(
        {
            "figure.facecolor": background,
            "axes.facecolor": background,
            "savefig.facecolor": background,
            "text.color": foreground,
            "axes.labelcolor": foreground,
            "axes.edgecolor": grid,
            "xtick.color": foreground,
            "ytick.color": foreground,
            "grid.color": grid,
            "font.family": "serif",
            "font.size": 10,
        }
    )


def save_pair(name: str, draw: Callable[[bool], plt.Figure]) -> None:
    apply_style(False)
    figure = draw(False)
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)

    apply_style(True)
    figure = draw(True)
    figure.savefig(DARK_FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def trajectory_figure(dark: bool) -> plt.Figure:
    series = PILOT["representative_trajectory"]["series"]
    config = PILOT["config"]
    z = np.asarray(series["redshift"])
    expansion = np.asarray(series["expansion"])
    density = np.asarray(series["f_de"])
    interval_z = 0.5 * (z[1:] + z[:-1])
    interval_w = np.asarray(series["w_interval"])
    lcdm = np.sqrt(
        config["omega_m"] * (1.0 + z) ** 3
        + config["omega_r"] * (1.0 + z) ** 4
        + 1.0
        - config["omega_m"]
        - config["omega_r"]
    )
    accent = "#69a7ff" if dark else "#235fa4"
    orange = "#ff8b5c" if dark else "#b44a1b"
    gray = "#b7bac2" if dark else "#6a6d75"
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    axes[0].plot(z, expansion, color=accent, lw=2.2, label="DSCD+GR")
    axes[0].plot(z, lcdm, color=gray, lw=1.5, ls="--", label=r"$\Lambda$CDM")
    axes[0].set(xlabel="redshift z", ylabel=r"$E(z)=H(z)/H_0$")
    axes[0].legend(frameon=False)

    axes[1].plot(z, density, color=accent, lw=2.2)
    axes[1].set(xlabel="redshift z", ylabel=r"$\rho_C(z)/\rho_C(0)$")

    axes[2].plot(interval_z, interval_w, color=orange, lw=1.8)
    axes[2].axhline(-1.0, color=gray, lw=1.2, ls="--")
    axes[2].set(xlabel="redshift z", ylabel=r"interval $w_C$")

    for axis in axes:
        axis.grid(alpha=0.35)
        axis.invert_xaxis()
    figure.suptitle("Emergent background from one audited stochastic trajectory", y=1.03)
    figure.tight_layout()
    return figure


def score_figure(dark: bool) -> plt.Figure:
    releases = ("DESI_DR1", "DESI_DR2")
    models = ("DSCD_FIXED", "LCDM", "WCDM_NONPHANTOM", "CPL", "CPL_NONPHANTOM")
    labels = ("DSCD fixed", r"$\Lambda$CDM", r"$w\geq-1$", "CPL", "CPL non-phantom")
    colors = ["#69a7ff", "#9fa3ad", "#77c593", "#ff8b5c", "#d5a6ff"] if dark else [
        "#235fa4",
        "#6a6d75",
        "#2c8050",
        "#b44a1b",
        "#74439b",
    ]
    values = []
    for release in releases:
        row = []
        for model in models:
            result = RETROSPECTIVE["full_fits"][release][model]
            row.append(
                result["chi2_observational"]
                if model == "DSCD_FIXED"
                else result["fit"]["chi2"]
            )
        values.append(row)
    values_array = np.asarray(values)
    x = np.arange(len(releases))
    width = 0.15
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    for index, (label, color) in enumerate(zip(labels, colors)):
        offset = (index - 2) * width
        bars = axis.bar(
            x + offset,
            values_array[:, index],
            width=width,
            label=label,
            color=color,
        )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axis.set_xticks(x, ("DR1", "DR2"))
    axis.set_ylabel(r"retrospective full-covariance $\chi^2$")
    axis.set_title("Background-distance fits (descriptive, not model probabilities)")
    axis.grid(axis="y", alpha=0.35)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    return figure


def audit_figure(dark: bool) -> plt.Figure:
    ablations = AUDIT["summary"]["DR2_ablation_squared_separations"]
    labels = [
        "no memory",
        "no regimes",
        "no transport",
        "no XX depletion",
        "zero beat",
    ]
    keys = [
        "no_memory",
        "no_regime_feedback",
        "no_transport",
        "symmetric_depletion",
        "zero_beat",
    ]
    values = np.asarray([ablations[key] for key in keys])
    accent = "#69a7ff" if dark else "#235fa4"
    threshold = "#ff8b5c" if dark else "#b44a1b"
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.barh(labels, values, color=accent)
    axis.axvline(1.0, color=threshold, lw=1.8, ls="--", label="materiality gate = 1")
    axis.set_xscale("log")
    axis.set_xlabel(r"squared observational separation $d^2$")
    axis.set_title("Defining DSCD ablations are unresolved by current compressed BAO")
    axis.grid(axis="x", which="both", alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    DARK_FIGURES.mkdir(parents=True, exist_ok=True)
    save_pair("dscd_cosmology_trajectory", trajectory_figure)
    save_pair("dscd_cosmology_scores", score_figure)
    save_pair("dscd_cosmology_audit", audit_figure)
    print("Generated audited DSCD cosmology figures.")
