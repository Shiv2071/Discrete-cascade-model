"""Shared configuration, provenance, serialization, and reporting helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import scipy

from desi_bao_likelihood import BAODataset, PINNED_COMMIT
from desi_cosmology_models import DEFAULT_MODEL_ORDER, MODELS, FitResult, ModelSpec, fit_model


SCHEMA_VERSION = "clean-desi-results-v1"
ANALYSIS_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "clean_desi_results"

MODE_SETTINGS = {
    "quick": {
        "quadrature_order": 48,
        "seeds": [17, 43],
        "optimizer_maxiter": 70,
        "profile_points": 9,
        "bootstrap_replicates": 8,
    },
    "publication": {
        "quadrature_order": 96,
        "seeds": [17, 43, 101],
        "optimizer_maxiter": 180,
        "profile_points": 15,
        "bootstrap_replicates": 24,
    },
}

ASSUMPTIONS = [
    "Retrospective analysis of already-public DESI DR1/DR2 compressed BAO data.",
    "Gaussian BAO likelihood with covariance treated as fixed in cosmology.",
    "Spatially flat FRW background.",
    "BAO scale parameter theta=H0*r_d/c; H0 and r_d are not separately inferred.",
    "No CMB, supernova, or external sound-horizon prior.",
    "Baseline radiation density is zero; a fixed-radiation sensitivity is reported separately.",
]


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS), default="quick")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODEL_ORDER,
        default=list(DEFAULT_MODEL_ORDER),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    names = [
        "desi_bao_likelihood.py",
        "desi_cosmology_models.py",
        "desi_analysis_common.py",
    ]
    return {name: file_sha256(HERE / name) for name in names if (HERE / name).exists()}


def environment_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def base_payload(
    protocol: str,
    mode: str,
    runner_file: str,
    selected_models: Iterable[str] | None = None,
) -> dict[str, Any]:
    configuration = {
        "mode": mode,
        **MODE_SETTINGS[mode],
        "model_order": list(DEFAULT_MODEL_ORDER),
        "selected_models": list(selected_models or DEFAULT_MODEL_ORDER),
        "baseline_omega_r": 0.0,
        "radiation_sensitivity_omega_r": 9.0e-5,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "mode": mode,
        "configuration": configuration,
        "configuration_sha256": canonical_hash(configuration),
        "runner": runner_file,
        "runner_sha256": file_sha256(HERE / runner_file),
        "source_sha256": source_hashes(),
        "environment": environment_manifest(),
        "official_data_commit": PINNED_COMMIT,
        "assumptions": list(ASSUMPTIONS),
        "scientific_warnings": [],
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("JSON output contains NaN or Infinity")
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json_atomic(payload: dict[str, Any], destination: Path, force: bool) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to replace it")
    clean = sanitize_json(payload)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(clean, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def dataset_metadata(dataset: BAODataset) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "size": dataset.size,
        "labels": dataset.labels,
        "values": dataset.values.tolist(),
        "covariance": dataset.covariance.tolist(),
        "covariance_condition_number": float(np.linalg.cond(dataset.covariance)),
        "source": dataset.source,
    }


def fit_all_models(
    dataset: BAODataset,
    model_names: Iterable[str],
    mode: str,
    *,
    omega_r: float = 0.0,
    seeds: list[int] | None = None,
    maxiter: int | None = None,
) -> dict[str, FitResult]:
    settings = MODE_SETTINGS[mode]
    results: dict[str, FitResult] = {}
    for name in model_names:
        print(f"  fitting {name} to {dataset.name} ...", flush=True)
        results[name] = fit_model(
            MODELS[name],
            dataset,
            mode=mode,
            omega_r=omega_r,
            quadrature_order=settings["quadrature_order"],
            seeds=seeds,
            maxiter=maxiter,
        )
    return results


def information_criteria(fit: FitResult, model: ModelSpec, n: int) -> dict[str, Any]:
    k = model.parameter_count
    reasons: list[str] = []
    if not fit.regular_interior_fit:
        reasons.append("fit is boundary-constrained, singular, or not full-rank")
    if n <= k + 1:
        reasons.append("AICc requires n > k + 1")
    if reasons:
        return {"available": False, "reasons": reasons}
    aic = fit.chi2 + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    return {
        "available": True,
        "descriptive_only": True,
        "aic": float(aic),
        "aicc": float(aicc),
        "n": int(n),
        "k": int(k),
    }


def fit_report(
    dataset: BAODataset,
    model: ModelSpec,
    fit: FitResult,
    *,
    omega_r: float,
    quadrature_order: int,
) -> dict[str, Any]:
    prediction = model.predict(dataset, fit.parameters, omega_r, quadrature_order)
    report = fit.to_dict(model)
    report["fit_diagnostics"] = dataset.diagnostics(prediction)
    report["information_criteria"] = information_criteria(fit, model, dataset.size)
    report["w_at_reference_redshifts"] = {
        str(redshift): float(model.w(np.asarray([redshift]), fit.parameters)[0])
        for redshift in (0.0, 0.5, 1.0, 2.33)
    }
    return report


def widened_model(model: ModelSpec) -> ModelSpec:
    widened: list[tuple[float, float]] = []
    for name, (lower, upper) in zip(model.parameter_names, model.bounds):
        if name == "theta":
            widened.append((max(1.0e-4, lower * 0.75), upper * 1.25))
        elif name == "omega_m":
            widened.append((max(1.0e-4, lower * 0.5), min(0.95, upper * 1.25)))
        elif name == "w0" and lower == -1.0:
            widened.append((-1.0, upper + 0.5))
        elif name == "gamma":
            widened.append((0.0, upper * 2.0))
        else:
            width = upper - lower
            widened.append((lower - 0.5 * width, upper + 0.5 * width))
    return replace(model, bounds=tuple(widened))
