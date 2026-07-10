"""Generate white-paper and dark-website figures for the v2 DSCD forecast."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "model_simulation" / "dscd_v2_results"
FIGURES = ROOT / "figures"
DARK_FIGURES = FIGURES / "dark fig"


def load(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


FORECAST = load("forecast_production.json")
CALIBRATION = load("calibration.json")
AUDIT = load("audit_v2.json")


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


def forecast_interval_figure(dark: bool) -> plt.Figure:
    table = FORECAST["dr3_forecast"]["full"]
    labels = table["labels"]
    quantiles = np.asarray(table["quantiles"])
    dr2_values = np.asarray(FORECAST["data"]["DESI_DR2"]["values"])
    sigma = np.sqrt(np.diag(np.asarray(FORECAST["data"]["DESI_DR2"]["covariance"])))
    lcdm = np.asarray(FORECAST["baseline_overlays"]["LCDM"]["dr3_layout_prediction"])

    median = (quantiles[:, 3] - dr2_values) / sigma
    low_68 = (quantiles[:, 2] - dr2_values) / sigma
    high_68 = (quantiles[:, 4] - dr2_values) / sigma
    low_95 = (quantiles[:, 1] - dr2_values) / sigma
    high_95 = (quantiles[:, 5] - dr2_values) / sigma
    lcdm_offset = (lcdm - dr2_values) / sigma

    accent = "#69a7ff" if dark else "#235fa4"
    orange = "#ff8b5c" if dark else "#b44a1b"
    gray = "#b7bac2" if dark else "#6a6d75"
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    axis.axhline(0.0, color=gray, lw=1.2, ls="--")
    axis.vlines(x, low_95, high_95, color=accent, lw=2.0, alpha=0.35, label="95% forecast")
    axis.vlines(x, low_68, high_68, color=accent, lw=5.0, alpha=0.8, label="68% forecast")
    axis.plot(x, median, "o", color=accent, ms=5, label="forecast median")
    axis.plot(x, lcdm_offset, "x", color=orange, ms=7, mew=2, label=r"best-fit $\Lambda$CDM")
    axis.set_xticks(x, [label.replace(":", "\n") for label in labels], fontsize=7)
    axis.set_ylabel(r"(prediction $-$ DR2) / $\sigma_{\rm DR2}$")
    axis.set_title(
        "Sealed DR3-layout forecast from history-compatible DSCD+GR realizations"
    )
    axis.grid(axis="y", alpha=0.35)
    axis.legend(frameon=False, ncol=4, fontsize=8)
    figure.tight_layout()
    return figure


def dynamics_band_figure(dark: bool) -> plt.Figure:
    grids = FORECAST["grids"]
    z_mid = np.asarray(grids["z_interval_midpoints"])
    z_points = np.asarray(grids["z_points"])
    w_band = np.asarray(FORECAST["w_interval_band"]["quantiles"])
    f_band = np.asarray(FORECAST["f_de_band"]["quantiles"])

    accent = "#69a7ff" if dark else "#235fa4"
    orange = "#ff8b5c" if dark else "#b44a1b"
    gray = "#b7bac2" if dark else "#6a6d75"
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    axes[0].fill_between(z_mid, w_band[:, 1], w_band[:, 5], color=accent, alpha=0.20, label="95%")
    axes[0].fill_between(z_mid, w_band[:, 2], w_band[:, 4], color=accent, alpha=0.45, label="68%")
    axes[0].plot(z_mid, w_band[:, 3], color=accent, lw=1.8, label="median")
    axes[0].axhline(-1.0, color=gray, lw=1.2, ls="--", label="w = -1  (cosmological constant)")
    axes[0].set(xlabel="redshift z", ylabel=r"interval $w_C(z)$")
    axes[0].set_title("Emergent equation of state (non-phantom by construction)")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].fill_between(z_points, f_band[:, 1], f_band[:, 5], color=orange, alpha=0.20)
    axes[1].fill_between(z_points, f_band[:, 2], f_band[:, 4], color=orange, alpha=0.45)
    axes[1].plot(z_points, f_band[:, 3], color=orange, lw=1.8)
    axes[1].axhline(1.0, color=gray, lw=1.2, ls="--")
    axes[1].set(xlabel="redshift z", ylabel=r"$\rho_C(z)/\rho_C(0)$")
    axes[1].set_title("History-compatible DSCD density band")

    for axis in axes:
        axis.grid(alpha=0.35)
        axis.invert_xaxis()
    figure.tight_layout()
    return figure


def calibration_figure(dark: bool) -> plt.Figure:
    per_truth = CALIBRATION["per_truth"]
    f68 = np.asarray([item["fraction_68"] for item in per_truth])
    f95 = np.asarray([item["fraction_95"] for item in per_truth])
    x = np.arange(len(per_truth))

    accent = "#69a7ff" if dark else "#235fa4"
    orange = "#ff8b5c" if dark else "#b44a1b"
    gray = "#b7bac2" if dark else "#6a6d75"
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.plot(x, f95, "s-", color=accent, label="95% interval coverage")
    axis.plot(x, f68, "o-", color=orange, label="68% interval coverage")
    axis.axhline(0.95, color=accent, lw=1.0, ls="--", alpha=0.7)
    axis.axhline(0.68, color=orange, lw=1.0, ls="--", alpha=0.7)
    overall = CALIBRATION["gates"]
    axis.axhline(
        overall["coverage_95"]["observed"], color=accent, lw=1.6, ls=":",
        label=f"overall 95%: {overall['coverage_95']['observed']:.2f}",
    )
    axis.axhline(
        overall["coverage_68"]["observed"], color=orange, lw=1.6, ls=":",
        label=f"overall 68%: {overall['coverage_68']['observed']:.2f}",
    )
    axis.set_xticks(x, [f"T{index + 1}" for index in x])
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("synthetic truth")
    axis.set_ylabel("fraction of DR3 observables covered")
    axis.set_title(
        f"End-to-end synthetic coverage calibration "
        f"(status: {CALIBRATION['status']})"
    )
    axis.grid(alpha=0.35)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    axis.text(
        0.99,
        0.02,
        f"audit: {AUDIT['status']}",
        transform=axis.transAxes,
        ha="right",
        color=gray,
        fontsize=8,
    )
    return figure


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    DARK_FIGURES.mkdir(parents=True, exist_ok=True)
    save_pair("dscd_v2_forecast_intervals", forecast_interval_figure)
    save_pair("dscd_v2_dynamics_band", dynamics_band_figure)
    save_pair("dscd_v2_calibration", calibration_figure)
    print("Generated v2 DSCD forecast figures.")
