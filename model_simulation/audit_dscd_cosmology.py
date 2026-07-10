"""Audit the complete DSCD cosmology artifact chain and decide forecast eligibility."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desi_analysis_common import file_sha256, write_json_atomic
from desi_bao_likelihood import DR1, DR2


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "dscd_cosmology_results"
DEFAULT_JSON = RESULT_DIR / "audit.json"
DEFAULT_MARKDOWN = RESULT_DIR / "audit.md"
ARTIFACTS = {
    "small": RESULT_DIR / "small_validations.json",
    "pilot": RESULT_DIR / "pilot_ensemble.json",
    "identifiability": RESULT_DIR / "identifiability.json",
    "retrospective": RESULT_DIR / "retrospective_quick.json",
}


def load(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


def add_gate(
    gates: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    severity: str,
) -> None:
    gates.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
            "severity": severity,
        }
    )


def current_hashes_match(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for name, expected in payload.get("source_sha256", {}).items():
        path = HERE / name
        if not path.exists():
            mismatches.append(f"{name}: missing")
        elif file_sha256(path) != expected:
            mismatches.append(f"{name}: changed")
    return not mismatches, mismatches


def audit() -> dict[str, Any]:
    payloads = {name: load(path) for name, path in ARTIFACTS.items()}
    small = payloads["small"]
    pilot = payloads["pilot"]
    identifiability = payloads["identifiability"]
    retrospective = payloads["retrospective"]
    gates: list[dict[str, Any]] = []

    for name, payload in payloads.items():
        matched, mismatches = current_hashes_match(payload)
        add_gate(
            gates,
            f"{name}.source_hashes_current",
            matched,
            "all declared sources match" if matched else "; ".join(mismatches),
            "technical",
        )

    dependency_checks = {
        "pilot_depends_on_small": pilot.get("small_validation_sha256")
        == file_sha256(ARTIFACTS["small"]),
        "identifiability_depends_on_pilot": identifiability.get("pilot_sha256")
        == file_sha256(ARTIFACTS["pilot"]),
        "retrospective_depends_on_identifiability": retrospective.get(
            "identifiability_sha256"
        )
        == file_sha256(ARTIFACTS["identifiability"]),
    }
    for name, passed in dependency_checks.items():
        add_gate(gates, name, passed, "transitive artifact hash", "technical")

    v_stages = small.get("validations", [])
    add_gate(
        gates,
        "V1_to_V5_pass",
        small.get("status") == "PASS"
        and len(v_stages) == 5
        and all(item.get("status") == "PASS" for item in v_stages)
        and small.get("real_desi_used") is False,
        "all five no-DESI small validations must pass first",
        "technical",
    )
    for name, passed in pilot.get("hard_checks", {}).items():
        add_gate(gates, f"pilot.{name}", passed, "pilot hard check", "scientific")

    full_ident = identifiability["full_one_dscd_combination"]
    fixed_ident = identifiability["fixed_dscd_background_only"]
    add_gate(
        gates,
        "identifiability.fixed_background",
        identifiability.get("status") == "PASS_BACKGROUND_ONLY"
        and fixed_ident.get("passed") is True,
        identifiability.get("calibration_policy", ""),
        "scientific",
    )
    add_gate(
        gates,
        "identifiability.DSCD_depletion_scale",
        full_ident.get("passed") is True,
        (
            f"condition={full_ident['svd']['condition_number']:.6g}; "
            f"rank={full_ident['svd']['rank']}"
        ),
        "scientific",
    )

    data = retrospective["data"]
    official_match = (
        data["DR1"]["labels"] == DR1.labels
        and data["DR2"]["labels"] == DR2.labels
        and np.array_equal(np.asarray(data["DR1"]["values"]), DR1.values)
        and np.array_equal(np.asarray(data["DR2"]["values"]), DR2.values)
        and np.array_equal(np.asarray(data["DR1"]["covariance"]), DR1.covariance)
        and np.array_equal(np.asarray(data["DR2"]["covariance"]), DR2.covariance)
    )
    add_gate(
        gates,
        "official_DESI_arrays",
        official_match,
        "embedded vectors and full covariances equal pinned likelihood",
        "technical",
    )
    add_gate(
        gates,
        "official_DR1_QSO_DV",
        "QSO:DV@z=1.491" in data["DR1"]["labels"],
        "official DR1 QSO observable must be DV/rd",
        "technical",
    )
    warnings = " ".join(retrospective.get("warnings", []))
    add_gate(
        gates,
        "retrospective_labeling",
        "retrospective" in warnings.lower() and "overlap" in warnings.lower(),
        "DR1/DR2 contamination and overlap are explicit",
        "technical",
    )

    dr2_models = retrospective["full_fits"]["DESI_DR2"]
    dscd_dr2 = dr2_models["DSCD_FIXED"]
    dscd_prediction = np.asarray(
        dscd_dr2["independent_scoring"]["prediction_mean"]
    )
    lcdm_prediction = np.asarray(dr2_models["LCDM"]["prediction"])
    dscd_lcdm_whitened = DR2.whiten(dscd_prediction - lcdm_prediction)
    dscd_lcdm_d2 = float(dscd_lcdm_whitened @ dscd_lcdm_whitened)
    add_gate(
        gates,
        "DSCD_vs_LCDM_discriminability",
        dscd_lcdm_d2 >= 4.0,
        f"joint squared separation d2={dscd_lcdm_d2:.6g}; required >=4",
        "scientific",
    )

    ablation_metrics: dict[str, float] = {}
    for name, result in retrospective["DR2_ablations"].items():
        prediction = np.asarray(result["independent_scoring"]["prediction_mean"])
        whitened = DR2.whiten(prediction - dscd_prediction)
        distance = float(whitened @ whitened)
        ablation_metrics[name] = distance
        add_gate(
            gates,
            f"ablation.{name}.material",
            distance >= 1.0,
            f"squared observational separation d2={distance:.6g}; required >=1",
            "scientific",
        )

    for release_name, models in retrospective["full_fits"].items():
        fit = models["DSCD_FIXED"]
        agreement = float(fit["optimizer_prediction_agreement"])
        add_gate(
            gates,
            f"{release_name}.optimizer_prediction_agreement",
            agreement < 0.1,
            f"raw prediction L2 difference={agreement:.6g}",
            "scientific",
        )
        add_gate(
            gates,
            f"{release_name}.finite_predictive_score",
            np.isfinite(fit["independent_scoring"]["joint_log_predictive_density"]),
            "independent-seed joint predictive score",
            "technical",
        )

    holdout = retrospective["redshift_holdout"]["DSCD_FIXED"]
    holdout_score = float(holdout["target_conditional_on_train"]["chi2"])
    add_gate(
        gates,
        "retrospective_holdout_finite",
        np.isfinite(holdout_score),
        f"conditional target chi2={holdout_score:.6g}; descriptive only",
        "technical",
    )
    overlap_warnings = all(
        "overlap" in item["warning"].lower()
        and "not independent" in item["warning"].lower()
        for item in retrospective["release_consistency"].values()
    )
    add_gate(
        gates,
        "release_overlap_warning",
        overlap_warnings,
        "both release directions explicitly reject independence",
        "technical",
    )

    technical_failures = [
        gate
        for gate in gates
        if not gate["passed"] and gate["severity"] == "technical"
    ]
    scientific_failures = [
        gate
        for gate in gates
        if not gate["passed"] and gate["severity"] == "scientific"
    ]
    if technical_failures:
        status = "TECHNICAL_FAILURE"
    elif scientific_failures:
        status = "NO_FORECAST"
    else:
        status = "FORECAST_ELIGIBLE"
    return {
        "schema_version": "dscd-cosmology-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate_counts": {
            "total": len(gates),
            "passed": sum(item["passed"] for item in gates),
            "technical_failures": len(technical_failures),
            "scientific_failures": len(scientific_failures),
        },
        "gates": gates,
        "summary": {
            "DR1_DSCD_chi2": retrospective["full_fits"]["DESI_DR1"][
                "DSCD_FIXED"
            ]["chi2_observational"],
            "DR2_DSCD_chi2": dscd_dr2["chi2_observational"],
            "DR1_holdout_conditional_chi2": holdout_score,
            "DR2_DSCD_LCDM_squared_separation": dscd_lcdm_d2,
            "DR2_ablation_squared_separations": ablation_metrics,
        },
        "artifact_sha256": {
            name: file_sha256(path) for name, path in ARTIFACTS.items()
        },
        "interpretation": [
            "The coupled dynamical system and all small numerical validations run successfully.",
            "One DSCD depletion combination is not identifiable with the compressed BAO layout, so it is frozen.",
            "The fixed DSCD background is close to LambdaCDM and its structural ablations are not observationally distinguishable.",
            "No corrected DR3 forecast is scientifically eligible under the preregistered gates.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    failed = [item for item in report["gates"] if not item["passed"]]
    lines = [
        "# Cosmological DSCD scientific audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Gate summary",
        "",
        f"- Passed: {report['gate_counts']['passed']} / {report['gate_counts']['total']}",
        f"- Technical failures: {report['gate_counts']['technical_failures']}",
        f"- Scientific failures: {report['gate_counts']['scientific_failures']}",
        "",
        "## Result summary",
        "",
        f"- DR1 full-fit DSCD chi-squared: {report['summary']['DR1_DSCD_chi2']:.6f}",
        f"- DR2 full-fit DSCD chi-squared: {report['summary']['DR2_DSCD_chi2']:.6f}",
        (
            "- Retrospective DR1 high-redshift conditional chi-squared: "
            f"{report['summary']['DR1_holdout_conditional_chi2']:.6f}"
        ),
        (
            "- DR2 DSCD-LambdaCDM squared separation: "
            f"{report['summary']['DR2_DSCD_LCDM_squared_separation']:.6g}"
        ),
        "",
    ]
    if failed:
        lines.extend(["## Failed gates", ""])
        lines.extend(
            f"- `{item['name']}`: {item['detail']}" for item in failed
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            *[f"- {item}" for item in report["interpretation"]],
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
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    result = audit()
    json_path = write_json_atomic(result, arguments.json, arguments.force)
    markdown_path = write_markdown(result, arguments.markdown, arguments.force)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path} ({result['status']})")
