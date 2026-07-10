"""History-compatibility scoring for the v2 DSCD forecast.

Each prior sample is a complete latent DSCD+GR realization.  This module
simulates it, profiles the BAO scale theta analytically (all compressed
BAO distances scale exactly as 1/theta), scores the realization against
the joint DR1+DR2 history with full per-release covariance plus the
trajectory Monte-Carlo covariance, and returns everything needed for
importance-weighted forward prediction.

DR1 and DR2 overlap observationally and no cross-release covariance is
published, so the joint score treats the releases as independent.  That
caveat is declared here, propagated to every output artifact, and probed
by the DR2-only sensitivity reported by the forecast runner.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from desi_bao_likelihood import DR1, DR2, BAODataset
from dscd_cosmology_ensemble import run_ensemble
from dscd_cosmology_observables import trajectory_prediction
from dscd_forecast_prior import (
    PRIOR_NAMES,
    THETA_BOUNDS,
    PriorSample,
    trajectory_seeds,
)


OVERLAP_CAVEAT = (
    "DR1 and DR2 overlap and no cross-release covariance is published; "
    "the joint history score treats releases as independent."
)
HISTORY_DATASETS: tuple[BAODataset, ...] = (DR1, DR2)

# Gauss-Hermite nodes used to propagate the analytic scale posterior into
# the prediction pool.  Every compressed BAO distance scales exactly as
# s = 1/theta, so with a flat prior on s the conditional posterior of s is
# Gaussian in closed form; profiling it instead was shown (calibration V6)
# to undercover by collapsing the common-mode scale uncertainty.
SCALE_NODES = 8


def _gaussian_log_density(
    dataset: BAODataset, mean: np.ndarray, seed_covariance: np.ndarray
) -> dict[str, float]:
    residual = mean - dataset.values
    total = dataset.covariance + seed_covariance
    cholesky = np.linalg.cholesky(total)
    whitened = np.linalg.solve(cholesky, residual)
    chi2_total = float(whitened @ whitened)
    sign, logdet = np.linalg.slogdet(total)
    if sign <= 0.0:
        raise np.linalg.LinAlgError("total predictive covariance not positive definite")
    return {
        "chi2_total": chi2_total,
        "chi2_observational": dataset.chi2(mean),
        "log_density": -0.5
        * (chi2_total + logdet + dataset.size * np.log(2.0 * np.pi)),
    }


def profile_theta(
    unit_means: Sequence[np.ndarray], datasets: Sequence[BAODataset]
) -> tuple[float, bool]:
    """Exact joint generalized-least-squares theta over several releases.

    Every prediction scales as 1/theta, so with s = 1/theta the joint
    chi-squared is quadratic in s and the optimum is analytic.
    """
    numerator = 0.0
    denominator = 0.0
    for mean, dataset in zip(unit_means, datasets):
        inverse = np.linalg.inv(dataset.covariance)
        numerator += float(mean @ inverse @ dataset.values)
        denominator += float(mean @ inverse @ mean)
    if denominator <= 0.0 or not np.isfinite(numerator / denominator):
        raise ValueError("theta profiling is degenerate for this realization")
    theta = 1.0 / (numerator / denominator)
    clipped = float(np.clip(theta, THETA_BOUNDS[0], THETA_BOUNDS[1]))
    return clipped, bool(clipped != theta)


def scale_posterior(
    unit_means: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
    inverses: Sequence[np.ndarray],
) -> tuple[float, float]:
    """Closed-form Gaussian posterior of the common scale s = 1/theta.

    With predictions exactly linear in s and a flat prior on s, the joint
    conditional posterior is N(B/A, 1/A) with A = sum m'C^-1 m and
    B = sum m'C^-1 y.
    """
    a = 0.0
    b = 0.0
    for mean, data, inverse in zip(unit_means, values, inverses):
        a += float(mean @ inverse @ mean)
        b += float(mean @ inverse @ data)
    if a <= 0.0 or not np.isfinite(b / a) or b / a <= 0.0:
        raise ValueError("scale posterior is degenerate for this realization")
    return b / a, 1.0 / a


def evaluate_sample(
    sample: PriorSample,
    *,
    replicates: int,
    seed_group: int = 0,
    quadrature_order: int = 48,
    datasets: Sequence[BAODataset] = HISTORY_DATASETS,
    dr3_layout_index: int | None = None,
) -> dict[str, Any]:
    """Simulate one latent realization and score it against history.

    Realizations whose density collapses to zero before the present (or
    that break any engine invariant) are physically incompatible with
    producing today's universe; they receive zero weight and are recorded
    with their failure reason rather than silently discarded.

    `datasets` defaults to the official DR1+DR2 history; the calibration
    gate substitutes synthetic releases with identical covariances.  The
    DR3 forward layout is taken from `dr3_layout_index` (default: the
    last dataset, which carries the DR2 tracer layout that DR3 will
    remeasure).
    """
    datasets = tuple(datasets)
    layout = (
        len(datasets) - 1 if dr3_layout_index is None else int(dr3_layout_index)
    )
    seeds = trajectory_seeds(sample.index, replicates, group_offset=seed_group)
    record: dict[str, Any] = {
        "index": int(sample.index),
        "parameters": sample.as_dict(),
        "depletion_scale": sample.depletion_scale,
        "seeds": list(seeds),
        "seed_group": int(seed_group),
    }
    try:
        config = sample.to_config()
        ensemble = run_ensemble(config, seeds, variance_semantics="physical")
        unit_predictions = [
            np.stack(
                [
                    trajectory_prediction(item, dataset, 1.0, quadrature_order)
                    for item in ensemble.trajectories
                ]
            )
            for dataset in datasets
        ]
        unit_means = [np.mean(block, axis=0) for block in unit_predictions]
        theta_profile, theta_clipped = profile_theta(unit_means, datasets)

        # Seed covariance at the profiled scale, then the closed-form scale
        # posterior using the seed-inflated covariance (one refinement pass).
        seed_covariances = []
        for dataset, block in zip(datasets, unit_predictions):
            scaled = block / theta_profile
            seed_covariances.append(
                np.atleast_2d(np.cov(scaled, rowvar=False, ddof=1))
                if scaled.shape[0] > 1
                else np.zeros_like(dataset.covariance)
            )
        total_inverses = [
            np.linalg.inv(dataset.covariance + seed_cov)
            for dataset, seed_cov in zip(datasets, seed_covariances)
        ]
        s_hat, s_var = scale_posterior(
            unit_means, [dataset.values for dataset in datasets], total_inverses
        )
        theta = float(np.clip(1.0 / s_hat, THETA_BOUNDS[0], THETA_BOUNDS[1]))
        theta_clipped = bool(theta != 1.0 / s_hat) or theta_clipped

        # Joint marginal log density over the common scale (flat prior on s):
        # the chi-squared at s_hat is the quadratic minimum, and the Gaussian
        # integral contributes +0.5*log(2*pi*s_var).
        log_density = 0.5 * np.log(2.0 * np.pi * s_var)
        per_release: dict[str, Any] = {}
        for dataset, block, seed_cov in zip(datasets, unit_predictions, seed_covariances):
            mean = np.mean(block, axis=0) * s_hat
            score = _gaussian_log_density(dataset, mean, seed_cov)
            log_density += score["log_density"]
            per_release[dataset.name] = {
                **score,
                "prediction_mean": mean.tolist(),
            }

        # Forward DR3 pool: per trajectory seed and per Gauss-Hermite node of
        # the scale posterior, keeping realization variance and the common-
        # mode scale uncertainty that profiling would collapse.
        gh_nodes, gh_weights = np.polynomial.hermite_e.hermegauss(SCALE_NODES)
        scale_values = s_hat + np.sqrt(s_var) * gh_nodes
        scale_weights = gh_weights / np.sum(gh_weights)
        positive = scale_values > 0.0
        scale_values = scale_values[positive]
        scale_weights = scale_weights[positive] / np.sum(scale_weights[positive])
        dr3_unit = unit_predictions[layout]
        dr3_predictions = (
            dr3_unit[:, None, :] * scale_values[None, :, None]
        )

        record.update(
            {
                "status": "OK",
                "theta": theta,
                "theta_clipped": theta_clipped,
                "scale_mean": float(s_hat),
                "scale_sigma": float(np.sqrt(s_var)),
                "closure_converged": all(
                    item.closure_converged for item in ensemble.trajectories
                ),
                "joint_log_density": float(log_density),
                "per_release": per_release,
                # DR3 remeasures the layout release's tracers; axes are
                # (trajectory seed, scale node, observable).
                "dr3_release": datasets[layout].name,
                "dr3_predictions": dr3_predictions.tolist(),
                "dr3_scale_weights": scale_weights.tolist(),
                "w_interval_mean": np.mean(
                    ensemble.stack("w_interval"), axis=0
                ).tolist(),
                "f_de_mean": np.mean(
                    np.stack([item.f_de for item in ensemble.trajectories]), axis=0
                ).tolist(),
                "rho_depletion_fraction": float(
                    1.0
                    - np.mean([item.rho_c[-1] / item.rho_c[0] for item in ensemble.trajectories])
                ),
            }
        )
    except (ValueError, ArithmeticError, OverflowError, np.linalg.LinAlgError) as exc:
        record.update(
            {
                "status": "FAILED",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return record


def with_values(dataset: BAODataset, values: Sequence[float], suffix: str) -> BAODataset:
    """Same observables and covariance, substituted (synthetic) mean vector."""
    return BAODataset(
        name=f"{dataset.name}{suffix}",
        observables=dataset.observables,
        values=np.asarray(values, dtype=float),
        covariance=np.asarray(dataset.covariance, dtype=float).copy(),
        source={**dataset.source, "synthetic_substitution": True},
    )


def evaluate_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entry point; rebuilds the sample from plain values."""
    sample = PriorSample(int(payload["index"]), np.asarray(payload["values"]))
    synthetic = payload.get("synthetic_values")
    if synthetic is None:
        datasets: Sequence[BAODataset] = HISTORY_DATASETS
    else:
        datasets = tuple(
            with_values(dataset, synthetic[dataset.name], "_synthetic")
            for dataset in HISTORY_DATASETS
        )
    return evaluate_sample(
        sample,
        replicates=int(payload["replicates"]),
        seed_group=int(payload.get("seed_group", 0)),
        quadrature_order=int(payload.get("quadrature_order", 48)),
        datasets=datasets,
    )


def importance_weights(records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Self-normalized importance weights from joint log predictive densities."""
    log_weights = np.full(len(records), -np.inf)
    for position, record in enumerate(records):
        if record.get("status") == "OK":
            log_weights[position] = float(record["joint_log_density"])
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError("no prior sample produced a valid history score")
    shifted = log_weights - np.max(log_weights[finite])
    weights = np.where(finite, np.exp(shifted), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("importance weights sum to zero")
    return weights / total


def release_only_weights(
    records: Sequence[dict[str, Any]], release_name: str
) -> np.ndarray:
    """Sensitivity weights conditioned on a single release."""
    log_weights = np.full(len(records), -np.inf)
    for position, record in enumerate(records):
        if record.get("status") == "OK":
            log_weights[position] = float(
                record["per_release"][release_name]["log_density"]
            )
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError(f"no valid sample for release {release_name}")
    shifted = log_weights - np.max(log_weights[finite])
    weights = np.where(finite, np.exp(shifted), 0.0)
    return weights / float(np.sum(weights))


def parameter_names() -> tuple[str, ...]:
    return PRIOR_NAMES
