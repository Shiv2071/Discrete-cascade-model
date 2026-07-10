"""Retrospective within-DR1 low-redshift to high-redshift holdout analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from desi_analysis_common import (
    MODE_SETTINGS,
    base_payload,
    dataset_metadata,
    fit_all_models,
    parse_common_args,
    widened_model,
    write_json_atomic,
)
from desi_bao_likelihood import BAODataset, DR1, conditional_target_diagnostics
from desi_cosmology_models import MODELS, fit_model


TRAIN_INDICES = tuple(i for i, observable in enumerate(DR1.observables) if observable.redshift <= 0.930)
TARGET_INDICES = tuple(i for i, observable in enumerate(DR1.observables) if observable.redshift > 0.930)
HOLDOUT_WARNING = (
    "This split was constructed after DR1 was public. It is a retrospective redshift "
    "holdout, not a prospective blind test."
)


def bootstrap_model(
    model_name: str,
    fitted_parameters: np.ndarray,
    observed_target_chi2: float,
    *,
    mode: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    settings = MODE_SETTINGS[mode]
    model = MODELS[model_name]
    rng = np.random.default_rng(seed)
    mean = model.predict(DR1, fitted_parameters, quadrature_order=settings["quadrature_order"])
    samples = rng.multivariate_normal(mean, DR1.covariance, size=replicates)
    parameter_samples: list[list[float]] = []
    target_scores: list[float] = []
    failures: list[str] = []

    print(f"    bootstrapping {model_name}: {replicates} refits ...", flush=True)
    for replicate, sample in enumerate(samples):
        synthetic_full = BAODataset(
            f"synthetic_DR1_{model_name}_{replicate}",
            DR1.observables,
            sample,
            DR1.covariance,
            {"simulation": "parametric bootstrap", "generating_model": model_name},
        )
        synthetic_train = synthetic_full.subset(TRAIN_INDICES, "synthetic_low_z")
        try:
            fit = fit_model(
                model,
                synthetic_train,
                mode="quick",
                quadrature_order=settings["quadrature_order"],
                seeds=[seed + replicate + 1],
                maxiter=35 if mode == "quick" else 55,
            )
            prediction = model.predict(
                synthetic_full,
                fit.parameters,
                quadrature_order=settings["quadrature_order"],
            )
            diagnostic = conditional_target_diagnostics(
                synthetic_full, prediction, TRAIN_INDICES, TARGET_INDICES
            )
            parameter_samples.append(fit.parameters.tolist())
            target_scores.append(float(diagnostic["chi2"]))
            if not fit.converged:
                failures.append(f"replicate {replicate}: optimizer convergence flag false")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(f"replicate {replicate}: {exc}")

    parameters_array = np.asarray(parameter_samples, dtype=float)
    scores_array = np.asarray(target_scores, dtype=float)
    intervals = {}
    if parameters_array.size:
        for i, parameter in enumerate(model.parameter_names):
            q16, q50, q84 = np.percentile(parameters_array[:, i], [16.0, 50.0, 84.0])
            intervals[parameter] = {
                "p16": float(q16),
                "median": float(q50),
                "p84": float(q84),
            }
    exceedances = int(np.sum(scores_array >= observed_target_chi2))
    empirical_tail = (
        float((exceedances + 1) / (scores_array.size + 1)) if scores_array.size else None
    )
    return {
        "method": "parametric bootstrap with full-covariance simulation and refitting",
        "requested_replicates": int(replicates),
        "successful_replicates": int(scores_array.size),
        "seed": int(seed),
        "parameter_intervals_16_50_84_percent": intervals,
        "target_chi2_samples": scores_array.tolist(),
        "observed_target_chi2": float(observed_target_chi2),
        "empirical_upper_tail_fraction_plus_one_corrected": empirical_tail,
        "failures": failures,
        "warning": "Finite-bootstrap diagnostic under the fitted model; not a preregistered p-value.",
    }


def run(mode: str, output_dir: Path, force: bool, model_names: list[str]) -> Path:
    settings = MODE_SETTINGS[mode]
    train = DR1.subset(TRAIN_INDICES, "DESI_DR1_low_z")
    target = DR1.subset(TARGET_INDICES, "DESI_DR1_high_z")
    payload = base_payload(
        "retrospective_redshift_holdout", mode, Path(__file__).name, model_names
    )
    payload["scientific_warnings"].append(HOLDOUT_WARNING)
    payload["split"] = {
        "rule": "train z<=0.930; target z>0.930",
        "train_indices": list(TRAIN_INDICES),
        "target_indices": list(TARGET_INDICES),
        "train_labels": train.labels,
        "target_labels": target.labels,
        "fixed_before_run": True,
    }
    payload["data"] = {
        "full_DR1": dataset_metadata(DR1),
        "train": dataset_metadata(train),
        "target": dataset_metadata(target),
    }

    fits = fit_all_models(train, model_names, mode)
    results: dict[str, object] = {}
    for model_index, name in enumerate(model_names):
        model = MODELS[name]
        fit = fits[name]
        print(f"  evaluating holdout {name} ...", flush=True)
        full_prediction = model.predict(
            DR1, fit.parameters, quadrature_order=settings["quadrature_order"]
        )
        train_prediction = full_prediction[list(TRAIN_INDICES)]
        target_prediction = full_prediction[list(TARGET_INDICES)]
        conditional = conditional_target_diagnostics(
            DR1, full_prediction, TRAIN_INDICES, TARGET_INDICES
        )
        wide_model = widened_model(model)
        print(f"    wider-bound sensitivity {name} ...", flush=True)
        wide_fit = fit_model(
            wide_model,
            train,
            mode=mode,
            quadrature_order=settings["quadrature_order"],
            seeds=[settings["seeds"][0]],
            maxiter=max(45, settings["optimizer_maxiter"] // 2),
        )
        wide_prediction = wide_model.predict(
            DR1, wide_fit.parameters, quadrature_order=settings["quadrature_order"]
        )
        wide_conditional = conditional_target_diagnostics(
            DR1, wide_prediction, TRAIN_INDICES, TARGET_INDICES
        )
        bootstrap = bootstrap_model(
            name,
            fit.parameters,
            float(conditional["chi2"]),
            mode=mode,
            replicates=settings["bootstrap_replicates"],
            seed=20260710 + 1000 * model_index,
        )
        results[name] = {
            "fit": fit.to_dict(model),
            "train": train.diagnostics(train_prediction),
            "target_marginal": target.diagnostics(target_prediction),
            "target_conditional_on_train": conditional,
            "wider_bounds_sensitivity": {
                "bounds": [list(bounds) for bounds in wide_model.bounds],
                "fit": wide_fit.to_dict(wide_model),
                "target_conditional_on_train": wide_conditional,
                "delta_target_chi2": float(wide_conditional["chi2"] - conditional["chi2"]),
                "unstable": bool(
                    wide_fit.active_bounds
                    or abs(float(wide_conditional["chi2"] - conditional["chi2"])) > 0.1
                ),
            },
            "bootstrap": bootstrap,
            "warning": HOLDOUT_WARNING,
        }
    payload["results"] = results

    destination = output_dir / f"redshift_holdout_{mode}.json"
    return write_json_atomic(payload, destination, force)


if __name__ == "__main__":
    arguments = parse_common_args(__doc__ or "Run retrospective redshift holdout.")
    result_path = run(arguments.mode, arguments.output_dir, arguments.force, arguments.models)
    print(f"\nWrote {result_path}")
