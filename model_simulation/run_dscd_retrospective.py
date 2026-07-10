"""Retrospective DR1/DR2 analysis of the fixed-dynamics DSCD+GR system."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desi_analysis_common import dataset_metadata, file_sha256, write_json_atomic
from desi_bao_likelihood import (
    DR1,
    DR2,
    BAODataset,
    conditional_target_diagnostics,
)
from desi_cosmology_models import MODELS, fit_model
from dscd_cosmology_config import DSCDCosmologyConfig, SYSTEM_VERSION
from dscd_cosmology_inference import (
    DSCDTrajectoryLikelihood,
    fit_fixed_dscd_background,
)


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "dscd_cosmology_results"
IDENTIFIABILITY_PATH = RESULT_DIR / "identifiability.json"
DEFAULT_OUTPUT = RESULT_DIR / "retrospective_quick.json"
TRAIN_INDICES = tuple(
    index
    for index, observable in enumerate(DR1.observables)
    if observable.redshift <= 0.930
)
TARGET_INDICES = tuple(
    index
    for index, observable in enumerate(DR1.observables)
    if observable.redshift > 0.930
)
TRAINING_SEEDS = (701, 709)
SCORING_SEEDS = (811, 821, 823, 827)
BASELINES = (
    "LCDM",
    "WCDM_NONPHANTOM",
    "CPL",
    "CPL_NONPHANTOM",
    "MONO_ONE",
)


def require_identifiability() -> dict[str, Any]:
    with IDENTIFIABILITY_PATH.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "PASS_BACKGROUND_ONLY":
        raise RuntimeError(
            "version 1 may proceed only under the frozen-DSCD calibration policy"
        )
    return report


def independent_dscd_score(
    config: DSCDCosmologyConfig,
    dataset: BAODataset,
    parameter_vector: list[float],
) -> dict[str, Any]:
    likelihood = DSCDTrajectoryLikelihood(
        config,
        dataset,
        SCORING_SEEDS,
        quadrature_order=48,
        physical_variance=True,
    )
    score = likelihood.score(parameter_vector, fit_depletion=False)
    return {
        "trajectory_seeds": list(SCORING_SEEDS),
        "chi2_observational": float(score["chi2_observational"]),
        "chi2_total_predictive": float(score["chi2_total"]),
        "joint_log_predictive_density": float(
            score["joint_log_predictive_density"]
        ),
        "prediction_mean": score["prediction_mean"].tolist(),
        "simulation_covariance": score["simulation_covariance"].tolist(),
        "raw_residual": score["raw_residual"].tolist(),
        "whitened_residual": score["whitened_residual"].tolist(),
    }


def fit_dscd_dataset(
    config: DSCDCosmologyConfig, dataset: BAODataset, profile_points: int = 21
) -> dict[str, Any]:
    fit = fit_fixed_dscd_background(
        config,
        dataset,
        TRAINING_SEEDS,
        quadrature_order=48,
        profile_points=profile_points,
    )
    fit["independent_scoring"] = independent_dscd_score(
        config,
        dataset,
        fit["parameter_vector"],
    )
    return fit


def baseline_report(dataset: BAODataset, model_name: str) -> dict[str, Any]:
    model = MODELS[model_name]
    fit = fit_model(
        model,
        dataset,
        mode="quick",
        quadrature_order=48,
        seeds=(17, 43),
        maxiter=45,
    )
    prediction = model.predict(dataset, fit.parameters, quadrature_order=48)
    return {
        "fit": fit.to_dict(model),
        "diagnostics": dataset.diagnostics(prediction),
        "prediction": prediction.tolist(),
    }


def run() -> dict[str, Any]:
    identifiability = require_identifiability()
    config = DSCDCosmologyConfig().validate()

    print("Fitting frozen DSCD background to DR1...", flush=True)
    dscd_dr1 = fit_dscd_dataset(config, DR1)
    print("Fitting frozen DSCD background to DR2...", flush=True)
    dscd_dr2 = fit_dscd_dataset(config, DR2)
    print("Fitting analytic control models...", flush=True)
    baselines = {
        dataset.name: {
            name: baseline_report(dataset, name) for name in BASELINES
        }
        for dataset in (DR1, DR2)
    }

    train = DR1.subset(TRAIN_INDICES, "DESI_DR1_low_z_retrospective")
    print("Running retrospective within-DR1 holdout...", flush=True)
    dscd_train = fit_dscd_dataset(config, train, profile_points=17)
    scoring_full = DSCDTrajectoryLikelihood(
        config,
        DR1,
        SCORING_SEEDS,
        quadrature_order=48,
        physical_variance=True,
    )
    holdout_prediction, holdout_simulation_covariance = (
        scoring_full.predictive_distribution(
            dscd_train["parameter_vector"], fit_depletion=False
        )
    )
    dscd_holdout = {
        "warning": (
            "Constructed after DR1 was public; retrospective stress test, "
            "not a blind prediction."
        ),
        "train_indices": list(TRAIN_INDICES),
        "target_indices": list(TARGET_INDICES),
        "fit": dscd_train,
        "full_prediction": holdout_prediction.tolist(),
        "simulation_covariance": holdout_simulation_covariance.tolist(),
        "target_marginal": DR1.subset(TARGET_INDICES).diagnostics(
            holdout_prediction[np.asarray(TARGET_INDICES)]
        ),
        "target_conditional_on_train": conditional_target_diagnostics(
            DR1,
            holdout_prediction,
            TRAIN_INDICES,
            TARGET_INDICES,
        ),
    }
    baseline_holdout: dict[str, Any] = {}
    for name in BASELINES:
        model = MODELS[name]
        fit = fit_model(
            model,
            train,
            mode="quick",
            quadrature_order=48,
            seeds=(17, 43),
            maxiter=45,
        )
        prediction = model.predict(DR1, fit.parameters, quadrature_order=48)
        baseline_holdout[name] = {
            "fit": fit.to_dict(model),
            "train": train.diagnostics(prediction[np.asarray(TRAIN_INDICES)]),
            "target_marginal": DR1.subset(TARGET_INDICES).diagnostics(
                prediction[np.asarray(TARGET_INDICES)]
            ),
            "target_conditional_on_train": conditional_target_diagnostics(
                DR1,
                prediction,
                TRAIN_INDICES,
                TARGET_INDICES,
            ),
        }

    release_directions: dict[str, Any] = {}
    for name, fit, source, target in (
        ("DR1_to_DR2", dscd_dr1, DR1, DR2),
        ("DR2_to_DR1", dscd_dr2, DR2, DR1),
    ):
        target_score = independent_dscd_score(
            config,
            target,
            fit["parameter_vector"],
        )
        release_directions[name] = {
            "warning": (
                "DR1 and DR2 overlap and cross-release covariance is unavailable; "
                "this is descriptive release consistency, not independent validation."
            ),
            "train_release": source.name,
            "target_release": target.name,
            "fit": fit,
            "target": target_score,
        }

    ablation_configs = {
        "zero_beat": replace(config, omega_y=config.omega_x).validate(),
        "no_memory": replace(
            config, gamma_xy=0.0, gamma_xx=0.0, gamma_bond=0.0
        ).validate(),
        "no_regime_feedback": replace(
            config, leakage_rate=0.0, explosion_rate=0.0
        ).validate(),
        "no_transport": replace(
            config, diffusion_x=0.0, diffusion_y=0.0
        ).validate(),
        "symmetric_depletion": replace(config, alpha_xx=0.0).validate(),
    }
    ablations: dict[str, Any] = {}
    for name, ablation in ablation_configs.items():
        print(f"Fitting DR2 ablation: {name}...", flush=True)
        ablations[name] = fit_dscd_dataset(ablation, DR2, profile_points=13)

    return {
        "schema_version": "dscd-retrospective-v1",
        "system_version": SYSTEM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_RETROSPECTIVE",
        "calibration_policy": identifiability["calibration_policy"],
        "warnings": [
            "Every DR1/DR2 result is retrospective.",
            "All DSCD dynamics are frozen; only theta and omega_m are fitted.",
            "DR1 and DR2 overlap; release-direction scores are descriptive.",
            "Compressed BAO tests background distances only.",
        ],
        "data": {"DR1": dataset_metadata(DR1), "DR2": dataset_metadata(DR2)},
        "config": config.to_dict(),
        "training_seeds": list(TRAINING_SEEDS),
        "independent_scoring_seeds": list(SCORING_SEEDS),
        "source_sha256": {
            name: file_sha256(HERE / name)
            for name in (
                "dscd_cosmology_dynamics.py",
                "dscd_cosmology_inference.py",
                "dscd_cosmology_observables.py",
                "desi_bao_likelihood.py",
                "desi_cosmology_models.py",
                "run_dscd_retrospective.py",
            )
        },
        "identifiability_sha256": file_sha256(IDENTIFIABILITY_PATH),
        "full_fits": {
            "DESI_DR1": {"DSCD_FIXED": dscd_dr1, **baselines["DESI_DR1"]},
            "DESI_DR2": {"DSCD_FIXED": dscd_dr2, **baselines["DESI_DR2"]},
        },
        "redshift_holdout": {
            "DSCD_FIXED": dscd_holdout,
            "baselines": baseline_holdout,
        },
        "release_consistency": release_directions,
        "DR2_ablations": ablations,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    report = run()
    destination = write_json_atomic(report, arguments.output, arguments.force)
    print(f"Wrote {destination} ({report['status']})")
