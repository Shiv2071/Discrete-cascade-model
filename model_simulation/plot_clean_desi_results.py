"""Generate clean DESI summary figures exclusively from validated result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desi_analysis_common import DEFAULT_OUTPUT_DIR, SCHEMA_VERSION, canonical_hash


def load_result(path: Path, expected_protocol: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema")
    if payload.get("protocol") != expected_protocol:
        raise ValueError(f"{path}: expected protocol {expected_protocol}")
    if canonical_hash(payload["configuration"]) != payload.get("configuration_sha256"):
        raise ValueError(f"{path}: configuration hash mismatch")
    return payload


def grouped_bars(
    axis: plt.Axes,
    groups: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(len(series), 1)
    for index, (label, values) in enumerate(series.items()):
        offset = (index - 0.5 * (len(series) - 1)) * width
        bars = axis.bar(x + offset, values, width=width, label=label, alpha=0.85)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
    axis.set_xticks(x)
    axis.set_xticklabels(groups, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=7)


def plot(input_dir: Path, output: Path, mode: str) -> Path:
    full = load_result(input_dir / f"full_fits_{mode}.json", "full_release_fits")
    release = load_result(
        input_dir / f"release_consistency_{mode}.json",
        "overlapping_release_consistency",
    )
    holdout = load_result(
        input_dir / f"redshift_holdout_{mode}.json",
        "retrospective_redshift_holdout",
    )
    configuration_hashes = {
        full["configuration_sha256"],
        release["configuration_sha256"],
        holdout["configuration_sha256"],
    }
    if len(configuration_hashes) != 1:
        raise ValueError("Input JSON files were produced with different configurations")

    model_names = list(full["results"]["DESI_DR1"])
    full_series = {
        release_name: [
            float(full["results"][release_name][name]["chi2"]) for name in model_names
        ]
        for release_name in ("DESI_DR1", "DESI_DR2")
    }
    consistency_series = {
        direction: [
            float(release["directions"][direction]["models"][name]["target"]["chi2"])
            for name in model_names
        ]
        for direction in ("DR1_to_DR2", "DR2_to_DR1")
    }
    holdout_values = [
        float(holdout["results"][name]["target_conditional_on_train"]["chi2"])
        for name in model_names
    ]
    bootstrap_medians = []
    for name in model_names:
        samples = np.asarray(
            holdout["results"][name]["bootstrap"]["target_chi2_samples"], dtype=float
        )
        bootstrap_medians.append(float(np.median(samples)) if samples.size else np.nan)

    figure, axes = plt.subplots(3, 1, figsize=(13, 15), constrained_layout=True)
    grouped_bars(
        axes[0],
        model_names,
        full_series,
        "Full-release in-sample chi-squared (retrospective)",
        "chi-squared",
    )
    grouped_bars(
        axes[1],
        model_names,
        consistency_series,
        "Overlapping-release consistency scores (not independent predictions)",
        "target chi-squared",
    )
    grouped_bars(
        axes[2],
        model_names,
        {
            "observed conditional target": holdout_values,
            "bootstrap median": bootstrap_medians,
        },
        "Within-DR1 retrospective redshift holdout",
        "conditional target chi-squared (log scale)",
    )
    axes[2].set_yscale("log")
    figure.suptitle(
        "Clean DESI BAO analysis\n"
        f"mode={mode}, configuration={next(iter(configuration_hashes))[:12]}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "publication"), default="quick")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    output_path = arguments.output or (
        arguments.input_dir / f"clean_desi_summary_{arguments.mode}.png"
    )
    print(f"Wrote {plot(arguments.input_dir, output_path, arguments.mode)}")
