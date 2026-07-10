"""Run the v2 history-compatible DSCD ensemble forecast for DESI DR3.

The pipeline samples latent DSCD+GR realizations from the declared prior,
scores each against the DR1+DR2 history, and propagates the importance-
weighted ensemble forward to the DR3 tracer layout.  The question answered
is predictive convergence: do history-compatible realizations funnel into
finite, stable DR3 intervals?  Parameter-level identifiability is neither
assumed nor required.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from desi_analysis_common import (
    dataset_metadata,
    environment_manifest,
    file_sha256,
    write_json_atomic,
)
from desi_bao_likelihood import DR2
from desi_cosmology_models import MODELS, fit_model
from dscd_cosmology_config import DSCDCosmologyConfig, SYSTEM_VERSION
from dscd_forecast_prior import (
    FORECAST_VERSION,
    PRIOR_NAMES,
    effective_sample_size,
    prior_metadata,
    sample_prior,
    weighted_quantiles,
)
from dscd_history_ensemble import (
    HISTORY_DATASETS,
    OVERLAP_CAVEAT,
    evaluate_sample_payload,
    importance_weights,
    release_only_weights,
)


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "dscd_v2_results"
ENGINE_STRESS_PATH = RESULT_DIR / "engine_stress.json"

SAMPLER_SEED = 20260710
QUANTILES = (0.005, 0.025, 0.16, 0.50, 0.84, 0.975, 0.995)
MODES = {
    "quick": {"samples": 256, "replicates": 4, "quadrature_order": 48},
    "production": {"samples": 2048, "replicates": 4, "quadrature_order": 48},
}
BASELINE_MODELS = ("LCDM", "CPL")
SOURCE_FILES = (
    "dscd_forecast_prior.py",
    "dscd_history_ensemble.py",
    "dscd_cosmology_config.py",
    "dscd_cosmology_dynamics.py",
    "dscd_cosmology_state.py",
    "dscd_cosmology_ensemble.py",
    "dscd_cosmology_observables.py",
    "dscd_cosmology_inference.py",
    "desi_bao_likelihood.py",
    "desi_cosmology_models.py",
    "run_dscd_v2_forecast.py",
)


def require_engine_stress() -> dict[str, Any]:
    import json

    with ENGINE_STRESS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "PASS":
        raise RuntimeError("V6a engine stress gate has not passed; forecast blocked")
    return payload


def evaluate_all(
    values: np.ndarray,
    *,
    replicates: int,
    quadrature_order: int,
    workers: int,
    synthetic_values: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    payloads = [
        {
            "index": index,
            "values": row.tolist(),
            "replicates": replicates,
            "seed_group": 0,
            "quadrature_order": quadrature_order,
            "synthetic_values": synthetic_values,
        }
        for index, row in enumerate(values)
    ]
    if workers <= 1:
        results = []
        for position, payload in enumerate(payloads):
            results.append(evaluate_sample_payload(payload))
            if (position + 1) % 64 == 0:
                print(f"  evaluated {position + 1}/{len(payloads)} samples", flush=True)
        return results
    results_ordered: list[dict[str, Any] | None] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        completed = 0
        for record in pool.map(evaluate_sample_payload, payloads, chunksize=8):
            results_ordered[record["index"]] = record
            completed += 1
            if completed % 64 == 0:
                print(f"  evaluated {completed}/{len(payloads)} samples", flush=True)
    return [item for item in results_ordered if item is not None]


def pooled_predictions(
    records: Sequence[dict[str, Any]],
    weights: np.ndarray,
    *,
    seed_slice: slice = slice(None),
) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-(sample, seed, scale-node) DR3 vectors with sample weights."""
    blocks: list[np.ndarray] = []
    block_weights: list[np.ndarray] = []
    for record, weight in zip(records, weights):
        if record.get("status") != "OK" or weight <= 0.0:
            continue
        block = np.asarray(record["dr3_predictions"], dtype=float)[seed_slice]
        scale_weights = np.asarray(record["dr3_scale_weights"], dtype=float)
        n_seeds = block.shape[0]
        flattened = block.reshape(-1, block.shape[-1])
        per_row = np.repeat(scale_weights[None, :], n_seeds, axis=0).reshape(-1)
        blocks.append(flattened)
        block_weights.append(weight * per_row / n_seeds)
    if not blocks:
        raise ValueError("no weighted predictions available")
    return np.concatenate(blocks, axis=0), np.concatenate(block_weights)


def interval_table(
    records: Sequence[dict[str, Any]],
    weights: np.ndarray,
    labels: Sequence[str],
    *,
    seed_slice: slice = slice(None),
) -> dict[str, Any]:
    pooled, pooled_weights = pooled_predictions(records, weights, seed_slice=seed_slice)
    table: dict[str, Any] = {"labels": list(labels), "quantile_levels": list(QUANTILES)}
    quantile_matrix = np.stack(
        [
            weighted_quantiles(pooled[:, column], pooled_weights, QUANTILES)
            for column in range(pooled.shape[1])
        ]
    )
    mean = np.average(pooled, axis=0, weights=pooled_weights)
    table["quantiles"] = quantile_matrix.tolist()
    table["mean"] = mean.tolist()
    table["interval_68_width"] = (quantile_matrix[:, 4] - quantile_matrix[:, 2]).tolist()
    table["interval_95_width"] = (quantile_matrix[:, 5] - quantile_matrix[:, 1]).tolist()
    return table


def series_band(
    records: Sequence[dict[str, Any]],
    weights: np.ndarray,
    field: str,
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    row_weights: list[float] = []
    for record, weight in zip(records, weights):
        if record.get("status") != "OK" or weight <= 0.0:
            continue
        rows.append(np.asarray(record[field], dtype=float))
        row_weights.append(float(weight))
    matrix = np.stack(rows)
    weight_vector = np.asarray(row_weights)
    band = np.stack(
        [
            weighted_quantiles(matrix[:, column], weight_vector, QUANTILES)
            for column in range(matrix.shape[1])
        ]
    )
    return {"quantile_levels": list(QUANTILES), "quantiles": band.tolist()}


def parameter_posterior(
    records: Sequence[dict[str, Any]], weights: np.ndarray
) -> dict[str, Any]:
    posterior: dict[str, Any] = {}
    ok = [
        (record, weight)
        for record, weight in zip(records, weights)
        if record.get("status") == "OK" and weight > 0.0
    ]
    weight_vector = np.asarray([weight for _, weight in ok])
    for name in PRIOR_NAMES:
        values = np.asarray([record["parameters"][name] for record, _ in ok])
        posterior[name] = weighted_quantiles(
            values, weight_vector, (0.16, 0.5, 0.84)
        ).tolist()
    extras = {
        "depletion_scale": np.asarray([record["depletion_scale"] for record, _ in ok]),
        "theta": np.asarray([record["theta"] for record, _ in ok]),
        "rho_depletion_fraction": np.asarray(
            [record["rho_depletion_fraction"] for record, _ in ok]
        ),
    }
    for name, values in extras.items():
        posterior[name] = weighted_quantiles(
            values, weight_vector, (0.16, 0.5, 0.84)
        ).tolist()
    return posterior


def baseline_overlays(quadrature_order: int) -> dict[str, Any]:
    overlays: dict[str, Any] = {}
    for name in BASELINE_MODELS:
        model = MODELS[name]
        fit = fit_model(
            model,
            DR2,
            mode="quick",
            quadrature_order=quadrature_order,
            seeds=(17, 43),
            maxiter=60,
        )
        prediction = model.predict(DR2, fit.parameters, quadrature_order=quadrature_order)
        overlays[name] = {
            "fit": fit.to_dict(model),
            "dr3_layout_prediction": prediction.tolist(),
            "chi2_dr2": DR2.chi2(prediction),
        }
    return overlays


def redshift_grids() -> dict[str, list[float]]:
    config = DSCDCosmologyConfig()
    steps = int(np.ceil(-config.n_start / config.dN))
    n_grid = np.linspace(config.n_start, 0.0, steps + 1)
    z_points = np.exp(-n_grid) - 1.0
    z_midpoints = np.exp(-0.5 * (n_grid[1:] + n_grid[:-1])) - 1.0
    return {
        "z_points": z_points.tolist(),
        "z_interval_midpoints": z_midpoints.tolist(),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in records if item.get("status") == "OK"]
    failed = [item for item in records if item.get("status") != "OK"]
    reasons: dict[str, int] = {}
    for item in failed:
        key = item.get("failure_reason", "unknown").split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "total": len(records),
        "completed": len(ok),
        "failed": len(failed),
        "failure_reasons": reasons,
        "theta_clipped": sum(bool(item.get("theta_clipped")) for item in ok),
    }


def run(mode: str, workers: int) -> dict[str, Any]:
    settings = MODES[mode]
    stress = require_engine_stress()
    samples = sample_prior(settings["samples"], SAMPLER_SEED)
    values = np.stack([item.values for item in samples])

    print(
        f"Evaluating {settings['samples']} prior samples "
        f"({settings['replicates']} seeds each, {workers} workers)...",
        flush=True,
    )
    records = evaluate_all(
        values,
        replicates=settings["replicates"],
        quadrature_order=settings["quadrature_order"],
        workers=workers,
    )
    weights = importance_weights(records)
    dr2_weights = release_only_weights(records, DR2.name)
    labels = DR2.labels

    print("Computing forecast intervals and gate inputs...", flush=True)
    half = len(records) // 2
    half_weights = importance_weights(records[:half])
    replicates = settings["replicates"]
    seed_first = slice(0, replicates // 2)
    seed_second = slice(replicates // 2, replicates)

    forecast = {
        "full": interval_table(records, weights, labels),
        "half_samples": interval_table(records[:half], half_weights, labels),
        "seed_split_a": interval_table(records, weights, labels, seed_slice=seed_first),
        "seed_split_b": interval_table(records, weights, labels, seed_slice=seed_second),
        "dr2_only_weights": interval_table(records, dr2_weights, labels),
    }
    dr2_sigma = np.sqrt(np.diag(DR2.covariance))
    forecast["width_to_dr2_sigma_68"] = (
        np.asarray(forecast["full"]["interval_68_width"]) / dr2_sigma
    ).tolist()

    return {
        "schema_version": "dscd-v2-forecast-v1",
        "forecast_version": FORECAST_VERSION,
        "system_version": SYSTEM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "COMPLETE",
        "question": (
            "Do history-compatible DSCD realizations converge on a finite, "
            "stable DR3 prediction?"
        ),
        "caveats": [
            OVERLAP_CAVEAT,
            "The common BAO scale theta is marginalized in closed form per "
            "realization (flat prior on 1/theta) and propagated through "
            "Gauss-Hermite nodes into every interval.",
            "Compressed BAO constrains background distances only.",
            "A forecast coincident with LambdaCDM remains a falsifiable forecast.",
        ],
        "prior": prior_metadata(),
        "sampler_seed": SAMPLER_SEED,
        "settings": settings,
        "base_config": DSCDCosmologyConfig().to_dict(),
        "data": {
            dataset.name: dataset_metadata(dataset) for dataset in HISTORY_DATASETS
        },
        "engine_stress_sha256": file_sha256(ENGINE_STRESS_PATH),
        "engine_stress_status": stress["status"],
        "source_sha256": {
            name: file_sha256(HERE / name) for name in SOURCE_FILES
        },
        "environment": environment_manifest(),
        "sample_summary": summarize(records),
        "effective_sample_size": {
            "joint": effective_sample_size(weights),
            "dr2_only": effective_sample_size(dr2_weights),
            "half_samples": effective_sample_size(half_weights),
        },
        "weight_concentration": {
            "max_weight": float(np.max(weights)),
            "top5_weight_mass": float(np.sum(np.sort(weights)[-5:])),
        },
        "dr3_forecast": forecast,
        "grids": redshift_grids(),
        "w_interval_band": series_band(records, weights, "w_interval_mean"),
        "f_de_band": series_band(records, weights, "f_de_mean"),
        "parameter_posterior": parameter_posterior(records, weights),
        "baseline_overlays": baseline_overlays(settings["quadrature_order"]),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), default="quick")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    arguments = parser.parse_args()
    report = run(arguments.mode, arguments.workers)
    destination = write_json_atomic(
        report,
        arguments.output or RESULT_DIR / f"forecast_{arguments.mode}.json",
        arguments.force,
    )
    print(f"Wrote {destination} ({report['status']})")
