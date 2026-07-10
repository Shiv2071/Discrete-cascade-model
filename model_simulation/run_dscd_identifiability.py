"""Synthetic identifiability gate for the trajectory-generated DSCD model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desi_analysis_common import file_sha256, write_json_atomic
from desi_bao_likelihood import BAODataset, DR2
from dscd_cosmology_config import DSCDCosmologyConfig, SYSTEM_VERSION
from dscd_cosmology_inference import (
    DSCDTrajectoryLikelihood,
    whitened_jacobian_svd,
)


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "dscd_cosmology_results"
PILOT_PATH = RESULT_DIR / "pilot_ensemble.json"
DEFAULT_OUTPUT = RESULT_DIR / "identifiability.json"


def _load_pilot() -> dict[str, Any]:
    with PILOT_PATH.open("r", encoding="utf-8") as handle:
        pilot = json.load(handle)
    if pilot.get("status") != "PASS" or pilot.get("real_desi_used") is not False:
        raise RuntimeError("pilot ensemble gate has not passed")
    return pilot


def fisher_uncertainty(svd_report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    whitened = np.asarray(svd_report["whitened_jacobian"], dtype=float)
    fisher = whitened.T @ whitened
    covariance = np.linalg.pinv(fisher, rcond=1.0e-10)
    return np.sqrt(np.diag(covariance)), covariance


def identifiability_report(
    likelihood: DSCDTrajectoryLikelihood,
    truth: np.ndarray,
    *,
    fit_depletion: bool,
) -> dict[str, Any]:
    svd = whitened_jacobian_svd(
        likelihood,
        truth,
        fit_depletion=fit_depletion,
    )
    uncertainty, covariance = fisher_uncertainty(svd)
    whitened = np.asarray(svd["whitened_jacobian"], dtype=float)
    fisher_inverse = np.linalg.pinv(whitened.T @ whitened, rcond=1.0e-10)
    estimator = fisher_inverse @ whitened.T
    rng = np.random.default_rng(20260710)
    whitened_noise = rng.normal(size=(1000, likelihood.dataset.size))
    recovered_delta = whitened_noise @ estimator.T
    recovered_mean = np.mean(recovered_delta, axis=0)
    recovered_std = np.std(recovered_delta, axis=0, ddof=1)
    normalized_bias = np.divide(
        np.abs(recovered_mean),
        uncertainty,
        out=np.full_like(recovered_mean, np.inf),
        where=uncertainty > 0.0,
    )
    coverage = np.mean(
        np.abs(recovered_delta) <= 1.959963984540054 * uncertainty,
        axis=0,
    )
    checks = {
        "full_rank": svd["rank"] == truth.size,
        "condition_below_1e4": svd["condition_number"] < 1.0e4,
        "normalized_bias_below_0p25": bool(np.all(normalized_bias < 0.25)),
        "linearized_95pct_coverage": bool(np.all((coverage > 0.925) & (coverage < 0.975))),
        "finite_uncertainty": bool(np.all(np.isfinite(uncertainty))),
    }
    return {
        "truth": truth.tolist(),
        "svd": svd,
        "fisher_standard_error": uncertainty.tolist(),
        "fisher_covariance": covariance.tolist(),
        "linearized_recovery_mean_delta": recovered_mean.tolist(),
        "linearized_recovery_std": recovered_std.tolist(),
        "normalized_bias": normalized_bias.tolist(),
        "linearized_95pct_coverage": coverage.tolist(),
        "synthetic_replicates": 1000,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run() -> dict[str, Any]:
    pilot = _load_pilot()
    config = DSCDCosmologyConfig().validate()
    layout = BAODataset(
        "synthetic_DSCD_identifiability_layout",
        DR2.observables,
        np.zeros(DR2.size),
        DR2.covariance,
        {
            "synthetic": True,
            "real_values_used": False,
            "covariance_layout": "DESI_DR2",
        },
    )
    truth_full = np.asarray([0.0332, 0.31, 1.4])
    generation = DSCDTrajectoryLikelihood(
        config,
        layout,
        (401, 409, 419, 421),
        quadrature_order=48,
        physical_variance=False,
    )
    synthetic_values, _ = generation.predictive_distribution(
        truth_full, fit_depletion=True
    )
    synthetic = BAODataset(
        "synthetic_DSCD_identifiability",
        layout.observables,
        synthetic_values,
        layout.covariance,
        layout.source,
    )
    inference_full = DSCDTrajectoryLikelihood(
        config,
        synthetic,
        (401, 409),
        quadrature_order=48,
        physical_variance=False,
    )
    full = identifiability_report(
        inference_full,
        truth_full,
        fit_depletion=True,
    )

    fixed_generation = DSCDTrajectoryLikelihood(
        config,
        layout,
        (401, 409),
        quadrature_order=48,
        physical_variance=False,
    )
    fixed_values, _ = fixed_generation.predictive_distribution(
        truth_full[:2], fit_depletion=False
    )
    fixed_synthetic = BAODataset(
        "synthetic_DSCD_fixed_dynamics",
        layout.observables,
        fixed_values,
        layout.covariance,
        layout.source,
    )
    inference_fixed = DSCDTrajectoryLikelihood(
        config,
        fixed_synthetic,
        (401, 409),
        quadrature_order=48,
        physical_variance=False,
    )
    fixed = identifiability_report(
        inference_fixed,
        truth_full[:2],
        fit_depletion=False,
    )

    independent = DSCDTrajectoryLikelihood(
        config,
        synthetic,
        (503, 509),
        quadrature_order=48,
        physical_variance=False,
    )
    independent_mean, _ = independent.predictive_distribution(
        truth_full, fit_depletion=True
    )
    independent_shift = synthetic.whiten(independent_mean - synthetic_values)
    independent_norm = float(np.linalg.norm(independent_shift))

    if full["passed"]:
        status = "PASS_ONE_DSCD_COMBINATION"
        calibration_policy = "Fit theta, omega_m, and one depletion_scale."
    elif fixed["passed"]:
        status = "PASS_BACKGROUND_ONLY"
        calibration_policy = (
            "Freeze all DSCD dynamics, including depletion_scale=1; "
            "fit only theta and omega_m."
        )
    else:
        status = "FAIL"
        calibration_policy = "No real-data DSCD fit is scientifically eligible."
    return {
        "schema_version": "dscd-identifiability-v1",
        "system_version": SYSTEM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "calibration_policy": calibration_policy,
        "real_desi_values_used": False,
        "real_desi_covariance_layout_used": True,
        "pilot_sha256": file_sha256(PILOT_PATH),
        "source_sha256": {
            name: file_sha256(HERE / name)
            for name in (
                "dscd_cosmology_inference.py",
                "dscd_cosmology_dynamics.py",
                "dscd_cosmology_observables.py",
                "run_dscd_identifiability.py",
            )
        },
        "synthetic": {
            "values": synthetic_values.tolist(),
            "labels": synthetic.labels,
            "generation_seeds": list(generation.trajectory_seeds),
        },
        "full_one_dscd_combination": full,
        "fixed_dscd_background_only": fixed,
        "independent_seed_whitened_shift_norm": independent_norm,
        "pilot_status": pilot["status"],
        "next_gate": (
            "retrospective_fit"
            if status.startswith("PASS")
            else "no_real_data_fit"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    report = run()
    destination = write_json_atomic(report, arguments.output, arguments.force)
    print(f"Wrote {destination} ({report['status']})")
