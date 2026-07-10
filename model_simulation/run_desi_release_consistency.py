"""Bidirectional DESI DR1/DR2 overlapping-release consistency diagnostics."""

from __future__ import annotations

from pathlib import Path

from desi_analysis_common import (
    MODE_SETTINGS,
    base_payload,
    dataset_metadata,
    fit_all_models,
    parse_common_args,
    write_json_atomic,
)
from desi_bao_likelihood import DR1, DR2
from desi_cosmology_models import MODELS


CROSS_RELEASE_WARNING = (
    "DR1 and DR2 overlap and no public aggregate DR1-DR2 cross-covariance is supplied. "
    "Target chi2 values are descriptive consistency scores only: they are not independent "
    "blind predictions, p-values, or a joint likelihood."
)


def run(mode: str, output_dir: Path, force: bool, model_names: list[str]) -> Path:
    settings = MODE_SETTINGS[mode]
    payload = base_payload(
        "overlapping_release_consistency", mode, Path(__file__).name, model_names
    )
    payload["scientific_warnings"].append(CROSS_RELEASE_WARNING)
    payload["data"] = {"DR1": dataset_metadata(DR1), "DR2": dataset_metadata(DR2)}
    fits_by_release = {
        "DR1": fit_all_models(DR1, model_names, mode),
        "DR2": fit_all_models(DR2, model_names, mode),
    }
    payload["directions"] = {}

    for train_label, train, target_label, target in (
        ("DR1", DR1, "DR2", DR2),
        ("DR2", DR2, "DR1", DR1),
    ):
        direction = f"{train_label}_to_{target_label}"
        models: dict[str, object] = {}
        for name in model_names:
            model = MODELS[name]
            fit = fits_by_release[train_label][name]
            train_prediction = model.predict(
                train, fit.parameters, quadrature_order=settings["quadrature_order"]
            )
            target_prediction = model.predict(
                target, fit.parameters, quadrature_order=settings["quadrature_order"]
            )
            models[name] = {
                "fit": fit.to_dict(model),
                "train": train.diagnostics(train_prediction),
                "target": target.diagnostics(target_prediction),
                "warning": CROSS_RELEASE_WARNING,
            }
        payload["directions"][direction] = {
            "train_release": train_label,
            "target_release": target_label,
            "models": models,
            "warning": CROSS_RELEASE_WARNING,
        }

    parameter_shifts: dict[str, object] = {}
    for name in model_names:
        model = MODELS[name]
        dr1_fit = fits_by_release["DR1"][name]
        dr2_fit = fits_by_release["DR2"][name]
        parameter_shifts[name] = {
            parameter: float(dr2_value - dr1_value)
            for parameter, dr1_value, dr2_value in zip(
                model.parameter_names, dr1_fit.parameters, dr2_fit.parameters
            )
        }
    payload["DR2_minus_DR1_best_fit_parameter_shifts"] = parameter_shifts

    destination = output_dir / f"release_consistency_{mode}.json"
    return write_json_atomic(payload, destination, force)


if __name__ == "__main__":
    arguments = parse_common_args(__doc__ or "Run release consistency diagnostics.")
    result_path = run(arguments.mode, arguments.output_dir, arguments.force, arguments.models)
    print(f"\nWrote {result_path}")
