"""V6: end-to-end coverage calibration of the v2 DSCD forecasting pipeline.

For synthetic truths drawn from the declared prior, the full pipeline is
run blind on synthetic DR1/DR2 realizations (official covariances, the
truth's own trajectory as the noiseless sky).  The gate checks that the
noiseless DR3-layout truth lands inside the predicted credible intervals
at the nominal rate.  A pipeline that cannot recover known truths at the
stated coverage is not allowed to issue a real forecast.

Caveat mirrored from the history score: synthetic DR1 and DR2 noise is
drawn independently, while the real releases overlap.  Real-data joint
conditioning is therefore anti-conservative; the DR2-only sensitivity in
the forecast artifact bounds that effect.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desi_analysis_common import file_sha256, write_json_atomic
from desi_bao_likelihood import DR1, DR2
from dscd_cosmology_dynamics import DSCDCosmologySystem
from dscd_cosmology_observables import trajectory_prediction
from dscd_forecast_prior import sample_prior
from run_dscd_v2_forecast import (
    ENGINE_STRESS_PATH,
    RESULT_DIR,
    evaluate_all,
    interval_table,
    require_engine_stress,
)
from dscd_history_ensemble import importance_weights


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = RESULT_DIR / "calibration.json"

TRUTH_COUNT = 12
TRUTH_CANDIDATES = 32
TRUTH_SAMPLER_SEED = 909_090
TRUTH_TRAJECTORY_SEED_BASE = 77_000_000
PIPELINE_SAMPLES = 512
PIPELINE_REPLICATES = 4
PIPELINE_SAMPLER_SEED_BASE = 505_000
NOISE_SEED_BASE = 313_000
QUADRATURE_ORDER = 48

# Nominal 68%/95% central intervals; thresholds allow for the finite number
# of checks (12 truths x 13 observables) and their within-truth correlation.
GATE_95_MINIMUM = 0.85
GATE_68_RANGE = (0.50, 0.86)


def synthetic_truth(candidate_index: int, sample: Any) -> dict[str, Any] | None:
    """One universe: a single stochastic trajectory plus a drawn theta."""
    rng = np.random.default_rng(NOISE_SEED_BASE + candidate_index)
    try:
        config = sample.to_config()
        trajectory = DSCDCosmologySystem(config).simulate(
            TRUTH_TRAJECTORY_SEED_BASE + candidate_index
        )
    except (ValueError, ArithmeticError, OverflowError) as exc:
        return {"status": "SKIPPED", "reason": f"{type(exc).__name__}: {exc}"}
    theta = float(rng.uniform(0.022, 0.042))
    noiseless = {
        DR1.name: trajectory_prediction(trajectory, DR1, theta, QUADRATURE_ORDER),
        DR2.name: trajectory_prediction(trajectory, DR2, theta, QUADRATURE_ORDER),
    }
    observed = {
        DR1.name: noiseless[DR1.name] + DR1.cholesky() @ rng.standard_normal(DR1.size),
        DR2.name: noiseless[DR2.name] + DR2.cholesky() @ rng.standard_normal(DR2.size),
    }
    return {
        "status": "OK",
        "candidate_index": candidate_index,
        "parameters": sample.as_dict(),
        "theta": theta,
        "trajectory_seed": TRUTH_TRAJECTORY_SEED_BASE + candidate_index,
        "noiseless": {name: values.tolist() for name, values in noiseless.items()},
        "observed": {name: values.tolist() for name, values in observed.items()},
        "dr3_truth": noiseless[DR2.name].tolist(),
    }


def coverage_flags(
    truth_vector: np.ndarray, quantiles: np.ndarray
) -> dict[str, list[bool]]:
    inside_68 = (truth_vector >= quantiles[:, 2]) & (truth_vector <= quantiles[:, 4])
    inside_95 = (truth_vector >= quantiles[:, 1]) & (truth_vector <= quantiles[:, 5])
    return {"inside_68": inside_68.tolist(), "inside_95": inside_95.tolist()}


def run(workers: int) -> dict[str, Any]:
    stress = require_engine_stress()
    candidates = sample_prior(TRUTH_CANDIDATES, TRUTH_SAMPLER_SEED)
    truths: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate_index, sample in enumerate(candidates):
        if len(truths) >= TRUTH_COUNT:
            break
        result = synthetic_truth(candidate_index, sample)
        if result["status"] == "OK":
            truths.append(result)
        else:
            skipped.append({"candidate_index": candidate_index, **result})
    if len(truths) < TRUTH_COUNT:
        raise RuntimeError("not enough completable synthetic truths on the prior")

    per_truth: list[dict[str, Any]] = []
    covered_68 = 0
    covered_95 = 0
    checked = 0
    for position, truth in enumerate(truths):
        print(
            f"Calibration truth {position + 1}/{TRUTH_COUNT} "
            f"(candidate {truth['candidate_index']})...",
            flush=True,
        )
        pipeline_samples = sample_prior(
            PIPELINE_SAMPLES, PIPELINE_SAMPLER_SEED_BASE + position
        )
        values = np.stack([item.values for item in pipeline_samples])
        records = evaluate_all(
            values,
            replicates=PIPELINE_REPLICATES,
            quadrature_order=QUADRATURE_ORDER,
            workers=workers,
            synthetic_values=truth["observed"],
        )
        weights = importance_weights(records)
        labels = DR2.labels
        table = interval_table(records, weights, labels)
        quantiles = np.asarray(table["quantiles"])
        truth_vector = np.asarray(truth["dr3_truth"])
        flags = coverage_flags(truth_vector, quantiles)
        covered_68 += int(np.sum(flags["inside_68"]))
        covered_95 += int(np.sum(flags["inside_95"]))
        checked += truth_vector.size
        ok = [item for item in records if item.get("status") == "OK"]
        per_truth.append(
            {
                "candidate_index": truth["candidate_index"],
                "truth_parameters": truth["parameters"],
                "truth_theta": truth["theta"],
                "pipeline_sampler_seed": PIPELINE_SAMPLER_SEED_BASE + position,
                "completed_samples": len(ok),
                "effective_sample_size": float(
                    np.sum(weights) ** 2 / np.sum(weights**2)
                ),
                "dr3_truth": truth["dr3_truth"],
                "interval_quantiles": table["quantiles"],
                "coverage": flags,
                "fraction_68": float(np.mean(flags["inside_68"])),
                "fraction_95": float(np.mean(flags["inside_95"])),
            }
        )

    overall_68 = covered_68 / checked
    overall_95 = covered_95 / checked
    pass_95 = overall_95 >= GATE_95_MINIMUM
    pass_68 = GATE_68_RANGE[0] <= overall_68 <= GATE_68_RANGE[1]
    return {
        "schema_version": "dscd-v2-calibration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if (pass_95 and pass_68) else "FAIL",
        "protocol": {
            "truth_count": TRUTH_COUNT,
            "truth_sampler_seed": TRUTH_SAMPLER_SEED,
            "pipeline_samples": PIPELINE_SAMPLES,
            "pipeline_replicates": PIPELINE_REPLICATES,
            "noise": "independent synthetic DR1/DR2 noise with official covariances",
            "target": "noiseless DR3-layout truth inside central credible intervals",
        },
        "gates": {
            "coverage_95": {
                "observed": overall_95,
                "required_minimum": GATE_95_MINIMUM,
                "passed": pass_95,
            },
            "coverage_68": {
                "observed": overall_68,
                "required_range": list(GATE_68_RANGE),
                "passed": pass_68,
            },
        },
        "checks_total": checked,
        "skipped_truth_candidates": skipped,
        "engine_stress_sha256": file_sha256(ENGINE_STRESS_PATH),
        "engine_stress_status": stress["status"],
        "source_sha256": {
            name: file_sha256(HERE / name)
            for name in (
                "dscd_forecast_prior.py",
                "dscd_history_ensemble.py",
                "run_dscd_v2_forecast.py",
                "validate_dscd_forecast_calibration.py",
            )
        },
        "per_truth": per_truth,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    arguments = parser.parse_args()
    report = run(arguments.workers)
    destination = write_json_atomic(report, arguments.output, arguments.force)
    print(f"Wrote {destination} ({report['status']})")
