"""Full in-sample DESI DR1/DR2 fits with profiles and sensitivity diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from desi_analysis_common import (
    MODE_SETTINGS,
    base_payload,
    dataset_metadata,
    fit_all_models,
    fit_report,
    parse_common_args,
    widened_model,
    write_json_atomic,
)
from desi_bao_likelihood import DR1, DR2
from desi_cosmology_models import MODELS, fit_model, profile_parameter


def goodness_of_fit(chi2: float, n: int, k: int, regular: bool) -> dict[str, object]:
    dof = n - k
    result: dict[str, object] = {
        "chi2": float(chi2),
        "nominal_dof": int(dof),
        "chi2_per_nominal_dof": float(chi2 / dof) if dof > 0 else None,
    }
    if regular and dof > 0:
        result["nominal_tail_probability"] = float(chi2_distribution.sf(chi2, dof))
        result["warning"] = "Nominal Gaussian goodness-of-fit diagnostic."
    else:
        result["nominal_tail_probability"] = None
        result["warning"] = "Suppressed because boundary/singularity invalidates naive degrees of freedom."
    return result


def run(mode: str, output_dir: Path, force: bool, model_names: list[str]) -> Path:
    settings = MODE_SETTINGS[mode]
    payload = base_payload("full_release_fits", mode, Path(__file__).name, model_names)
    payload["data"] = {"DR1": dataset_metadata(DR1), "DR2": dataset_metadata(DR2)}
    payload["results"] = {}
    payload["scientific_warnings"].append(
        "All fits are in-sample and retrospective; lower chi2 is not prospective validation."
    )

    for dataset in (DR1, DR2):
        print(f"\n{dataset.name}: full fit", flush=True)
        fits = fit_all_models(dataset, model_names, mode)
        release_results: dict[str, object] = {}
        for name, fit in fits.items():
            model = MODELS[name]
            print(f"    profiling {dataset.name} {name} ...", flush=True)
            report = fit_report(
                dataset,
                model,
                fit,
                omega_r=0.0,
                quadrature_order=settings["quadrature_order"],
            )
            report["goodness_of_fit"] = goodness_of_fit(
                fit.chi2, dataset.size, model.parameter_count, fit.regular_interior_fit
            )
            profile_name = "w0" if "w0" in model.parameter_names else "omega_m"
            report["likelihood_profile"] = profile_parameter(
                model,
                dataset,
                fit,
                profile_name,
                points=settings["profile_points"],
                quadrature_order=settings["quadrature_order"],
            )

            print(f"    sensitivity refits {dataset.name} {name} ...", flush=True)
            radiation_fit = fit_model(
                model,
                dataset,
                mode=mode,
                omega_r=9.0e-5,
                quadrature_order=settings["quadrature_order"],
                seeds=[settings["seeds"][0]],
                maxiter=max(45, settings["optimizer_maxiter"] // 2),
            )
            precision_fit = fit_model(
                model,
                dataset,
                mode=mode,
                quadrature_order=2 * settings["quadrature_order"],
                seeds=[settings["seeds"][0]],
                maxiter=max(45, settings["optimizer_maxiter"] // 2),
            )
            wide = widened_model(model)
            wide_fit = fit_model(
                wide,
                dataset,
                mode=mode,
                quadrature_order=settings["quadrature_order"],
                seeds=[settings["seeds"][0]],
                maxiter=max(45, settings["optimizer_maxiter"] // 2),
            )
            report["sensitivity"] = {
                "fixed_radiation": {
                    "omega_r": 9.0e-5,
                    "chi2": float(radiation_fit.chi2),
                    "delta_chi2_from_baseline": float(radiation_fit.chi2 - fit.chi2),
                    "parameters": radiation_fit.parameters.tolist(),
                    "active_bounds": radiation_fit.active_bounds,
                },
                "higher_quadrature": {
                    "quadrature_order": 2 * settings["quadrature_order"],
                    "chi2": float(precision_fit.chi2),
                    "delta_chi2_from_baseline": float(precision_fit.chi2 - fit.chi2),
                    "parameters": precision_fit.parameters.tolist(),
                    "active_bounds": precision_fit.active_bounds,
                },
                "wider_bounds": {
                    "bounds": [list(bounds) for bounds in wide.bounds],
                    "chi2": float(wide_fit.chi2),
                    "delta_chi2_from_baseline": float(wide_fit.chi2 - fit.chi2),
                    "parameters": wide_fit.parameters.tolist(),
                    "active_bounds": wide_fit.active_bounds,
                },
            }
            release_results[name] = report
        payload["results"][dataset.name] = release_results

    destination = output_dir / f"full_fits_{mode}.json"
    return write_json_atomic(payload, destination, force)


if __name__ == "__main__":
    arguments = parse_common_args(__doc__ or "Run full DESI fits.")
    result_path = run(arguments.mode, arguments.output_dir, arguments.force, arguments.models)
    print(f"\nWrote {result_path}")
