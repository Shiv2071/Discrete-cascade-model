"""Validate clean DESI result artifacts and produce a scientific gate report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from desi_analysis_common import (
    DEFAULT_OUTPUT_DIR,
    HERE,
    SCHEMA_VERSION,
    canonical_hash,
    file_sha256,
    source_hashes,
    write_json_atomic,
)
from desi_bao_likelihood import DR1, DR2


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant {value}")


def load(path: Path, protocol: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_constant)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema mismatch")
    if payload.get("protocol") != protocol:
        raise ValueError(f"{path}: protocol mismatch")
    if canonical_hash(payload["configuration"]) != payload["configuration_sha256"]:
        raise ValueError(f"{path}: configuration hash mismatch")
    return payload


def add_gate(
    gates: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    severity: str = "scientific",
) -> None:
    gates.append(
        {"name": name, "passed": bool(passed), "detail": detail, "severity": severity}
    )


def audit(input_dir: Path, mode: str) -> dict[str, Any]:
    full = load(input_dir / f"full_fits_{mode}.json", "full_release_fits")
    release = load(
        input_dir / f"release_consistency_{mode}.json",
        "overlapping_release_consistency",
    )
    holdout = load(
        input_dir / f"redshift_holdout_{mode}.json",
        "retrospective_redshift_holdout",
    )
    gates: list[dict[str, Any]] = []
    hashes = {
        full["configuration_sha256"],
        release["configuration_sha256"],
        holdout["configuration_sha256"],
    }
    add_gate(gates, "configuration_match", len(hashes) == 1, f"hashes={sorted(hashes)}", "technical")
    current_sources = source_hashes()
    source_match = all(
        payload.get("source_sha256") == current_sources
        for payload in (full, release, holdout)
    )
    add_gate(
        gates,
        "source_hashes_match_current_code",
        source_match,
        "All result artifacts must match the current likelihood/model/common source files.",
        "technical",
    )
    runner_match = all(
        payload.get("runner_sha256") == file_sha256(HERE / payload["runner"])
        for payload in (full, release, holdout)
    )
    add_gate(
        gates,
        "runner_hashes_match_current_code",
        runner_match,
        "Each result artifact must match its current runner source.",
        "technical",
    )
    add_gate(
        gates,
        "environment_manifests_match",
        full["environment"] == release["environment"] == holdout["environment"],
        "All canonical artifacts must use the same resolved environment.",
        "technical",
    )

    dr1_labels = full["data"]["DR1"]["labels"]
    add_gate(
        gates,
        "official_DR1_QSO_is_DV",
        "QSO:DV@z=1.491" in dr1_labels and not any(
            label.startswith("QSO:DH@z=1.491") for label in dr1_labels
        ),
        "Official DR1 QSO compressed observable must be DV/rd, never reconstructed DH/rd.",
        "technical",
    )
    official_data_match = (
        dr1_labels == DR1.labels
        and full["data"]["DR2"]["labels"] == DR2.labels
        and np.array_equal(np.asarray(full["data"]["DR1"]["values"]), DR1.values)
        and np.array_equal(np.asarray(full["data"]["DR2"]["values"]), DR2.values)
        and np.array_equal(
            np.asarray(full["data"]["DR1"]["covariance"]), DR1.covariance
        )
        and np.array_equal(
            np.asarray(full["data"]["DR2"]["covariance"]), DR2.covariance
        )
    )
    add_gate(
        gates,
        "embedded_official_arrays_match_likelihood",
        official_data_match,
        "Labels, vectors, and covariance matrices must equal the pinned in-code likelihood.",
        "technical",
    )
    add_gate(
        gates,
        "release_overlap_warning",
        all(
            "cross-covariance" in direction["warning"]
            for direction in release["directions"].values()
        ),
        "Every direction embeds the unavailable cross-release covariance warning.",
        "technical",
    )

    field_errors: list[str] = []
    for release_name, models in full["results"].items():
        expected_size = full["data"]["DR1" if release_name == "DESI_DR1" else "DR2"]["size"]
        for model_name, result in models.items():
            diagnostics = result["fit_diagnostics"]
            if abs(result["chi2"] - diagnostics["chi2"]) > 1.0e-9:
                field_errors.append(f"{release_name}.{model_name}: chi2 mismatch")
            for key in (
                "predictions",
                "raw_residuals",
                "marginal_standardized_residuals",
                "cholesky_whitened_residuals",
                "labels",
            ):
                if len(diagnostics[key]) != expected_size:
                    field_errors.append(f"{release_name}.{model_name}.{key}: wrong length")
            profile = result["likelihood_profile"]
            if not (
                len(profile["grid"])
                == len(profile["chi2"])
                == len(profile["delta_chi2"])
            ):
                field_errors.append(f"{release_name}.{model_name}: profile length mismatch")

    for direction_name, direction in release["directions"].items():
        for model_name, result in direction["models"].items():
            if abs(result["fit"]["chi2"] - result["train"]["chi2"]) > 1.0e-8:
                field_errors.append(f"{direction_name}.{model_name}: train chi2 mismatch")

    for model_name, result in holdout["results"].items():
        if abs(result["fit"]["chi2"] - result["train"]["chi2"]) > 1.0e-8:
            field_errors.append(f"holdout.{model_name}: train chi2 mismatch")
        conditional = result["target_conditional_on_train"]
        if conditional["cross_covariance_max_abs"] == 0.0 and abs(
            conditional["chi2"] - result["target_marginal"]["chi2"]
        ) > 1.0e-9:
            field_errors.append(f"holdout.{model_name}: zero-block reduction mismatch")
    add_gate(
        gates,
        "result_field_consistency",
        not field_errors,
        "all checked fields consistent" if not field_errors else "; ".join(field_errors),
        "technical",
    )

    summaries: dict[str, Any] = {"full_fits": {}, "release_targets": {}, "holdout": {}}
    for release_name, models in full["results"].items():
        summaries["full_fits"][release_name] = {}
        for model_name, result in models.items():
            summaries["full_fits"][release_name][model_name] = result["chi2"]
            seed_chi2 = [run["chi2"] for run in result["global_runs"]]
            seed_spread = max(seed_chi2) - min(seed_chi2)
            add_gate(
                gates,
                f"{release_name}.{model_name}.optimizer_agreement",
                seed_spread <= 1.0e-3,
                f"global seed chi2 spread={seed_spread:.6g}",
            )
            precision_delta = abs(
                result["sensitivity"]["higher_quadrature"]["delta_chi2_from_baseline"]
            )
            add_gate(
                gates,
                f"{release_name}.{model_name}.quadrature_stability",
                precision_delta <= 1.0e-3,
                f"|delta chi2|={precision_delta:.6g}",
            )
            wide = result["sensitivity"]["wider_bounds"]
            wide_delta = abs(wide["delta_chi2_from_baseline"])
            stable_bounds = not wide["active_bounds"] and wide_delta <= 0.1
            add_gate(
                gates,
                f"{release_name}.{model_name}.bound_stability",
                stable_bounds,
                f"|delta chi2|={wide_delta:.6g}; active={wide['active_bounds']}",
            )

    for direction_name, direction in release["directions"].items():
        summaries["release_targets"][direction_name] = {
            model_name: model_result["target"]["chi2"]
            for model_name, model_result in direction["models"].items()
        }

    for model_name, result in holdout["results"].items():
        bootstrap = result["bootstrap"]
        wide = result["wider_bounds_sensitivity"]
        summaries["holdout"][model_name] = {
            "train_chi2": result["train"]["chi2"],
            "target_conditional_chi2": result["target_conditional_on_train"]["chi2"],
            "bootstrap_tail_fraction": bootstrap[
                "empirical_upper_tail_fraction_plus_one_corrected"
            ],
            "wider_bounds_target_chi2": wide["target_conditional_on_train"]["chi2"],
        }
        add_gate(
            gates,
            f"holdout.{model_name}.bootstrap_completion",
            bootstrap["successful_replicates"] == bootstrap["requested_replicates"],
            (
                f"{bootstrap['successful_replicates']}/"
                f"{bootstrap['requested_replicates']} successful"
            ),
            "technical",
        )
        add_gate(
            gates,
            f"holdout.{model_name}.bound_stability",
            not wide["unstable"],
            (
                f"delta target chi2={wide['delta_target_chi2']:.6g}; "
                f"active={wide['fit']['active_bounds']}"
            ),
        )

    technical_failures = [
        gate for gate in gates if not gate["passed"] and gate["severity"] == "technical"
    ]
    scientific_failures = [
        gate for gate in gates if not gate["passed"] and gate["severity"] == "scientific"
    ]
    if technical_failures:
        status = "TECHNICAL_FAILURE"
    elif scientific_failures:
        status = "COMPLETE_WITH_SCIENTIFIC_LIMITATIONS"
    else:
        status = "PASS"
    return {
        "schema_version": "clean-desi-audit-v1",
        "mode": mode,
        "status": status,
        "configuration_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        "gate_counts": {
            "total": len(gates),
            "passed": sum(gate["passed"] for gate in gates),
            "technical_failures": len(technical_failures),
            "scientific_failures": len(scientific_failures),
        },
        "gates": gates,
        "summaries": summaries,
        "interpretation": [
            "In-sample chi2 values are descriptive and retrospective.",
            "DR1-to-DR2 and DR2-to-DR1 scores are not independent because releases overlap.",
            "Holdout bootstrap tails are finite-simulation diagnostics, not preregistered p-values.",
            "Any model failing optimizer, bound, or quadrature stability must not be ranked.",
            "No manuscript, website, historical seal, or Git history was modified by this analysis.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    failed = [gate for gate in report["gates"] if not gate["passed"]]
    lines = [
        "# Clean DESI numerical audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Mode: `{report['mode']}`",
        f"Configuration: `{report['configuration_sha256']}`",
        "",
        "## Gate summary",
        "",
        f"- Passed: {report['gate_counts']['passed']} / {report['gate_counts']['total']}",
        f"- Technical failures: {report['gate_counts']['technical_failures']}",
        f"- Scientific stability failures: {report['gate_counts']['scientific_failures']}",
        "",
    ]
    if failed:
        lines.extend(["## Failed stability gates", ""])
        lines.extend(
            f"- `{gate['name']}`: {gate['detail']}" for gate in failed
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation constraints",
            "",
            *[f"- {item}" for item in report["interpretation"]],
            "",
            "The complete numerical values, residuals, optimizer traces, profiles, and "
            "bootstrap samples remain in the accompanying strict JSON files.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write("\n".join(lines))
    os.replace(temporary, path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "publication"), default="quick")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    audit_report = audit(arguments.input_dir, arguments.mode)
    json_path = write_json_atomic(
        audit_report,
        arguments.input_dir / f"audit_{arguments.mode}.json",
        arguments.force,
    )
    markdown_path = write_markdown(
        audit_report,
        arguments.input_dir / f"audit_{arguments.mode}.md",
        arguments.force,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
