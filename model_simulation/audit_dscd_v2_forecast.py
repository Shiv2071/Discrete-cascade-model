"""Audit the v2 DSCD forecast chain and decide DR3 seal eligibility.

Version 2 gates on predictive convergence of the history-compatible
ensemble, not on parameter identifiability: internal DSCD parameters may
remain individually degenerate provided the forward DR3 prediction is
finite, stable, and produced by a pipeline whose coverage is calibrated.
"""

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
RESULT_DIR = HERE / "dscd_v2_results"
DEFAULT_JSON = RESULT_DIR / "audit_v2.json"
DEFAULT_MARKDOWN = RESULT_DIR / "audit_v2.md"
ARTIFACTS = {
    "engine_stress": RESULT_DIR / "engine_stress.json",
    "forecast": RESULT_DIR / "forecast_production.json",
    "calibration": RESULT_DIR / "calibration.json",
}

G1_MEDIAN_SHIFT_LIMIT = 0.10
G1_WIDTH_CHANGE_LIMIT = 0.25
G2_MEDIAN_SHIFT_LIMIT = 0.10
G3_MINIMUM_ESS = 25.0
G3_DR2_ONLY_SHIFT_LIMIT = 0.25


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
        {"name": name, "passed": bool(passed), "detail": detail, "severity": severity}
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


def interval_comparison(
    reference: dict[str, Any], alternate: dict[str, Any]
) -> tuple[float, float]:
    """Max |median shift| / 68-width and max |68-width ratio - 1|."""
    reference_q = np.asarray(reference["quantiles"])
    alternate_q = np.asarray(alternate["quantiles"])
    width = reference_q[:, 4] - reference_q[:, 2]
    shift = float(np.max(np.abs(alternate_q[:, 3] - reference_q[:, 3]) / width))
    alternate_width = alternate_q[:, 4] - alternate_q[:, 2]
    ratio = float(np.max(np.abs(alternate_width / width - 1.0)))
    return shift, ratio


def audit() -> dict[str, Any]:
    payloads = {name: load(path) for name, path in ARTIFACTS.items()}
    stress = payloads["engine_stress"]
    forecast = payloads["forecast"]
    calibration = payloads["calibration"]
    gates: list[dict[str, Any]] = []

    # Technical chain: source currency, artifact hash links, official data.
    for name, payload in payloads.items():
        matched, mismatches = current_hashes_match(payload)
        add_gate(
            gates,
            f"{name}.source_hashes_current",
            matched,
            "all declared sources match" if matched else "; ".join(mismatches),
            "technical",
        )
    stress_hash = file_sha256(ARTIFACTS["engine_stress"])
    add_gate(
        gates,
        "chain.forecast_depends_on_engine_stress",
        forecast.get("engine_stress_sha256") == stress_hash,
        "forecast recorded the audited V6a artifact hash",
        "technical",
    )
    add_gate(
        gates,
        "chain.calibration_depends_on_engine_stress",
        calibration.get("engine_stress_sha256") == stress_hash,
        "calibration recorded the audited V6a artifact hash",
        "technical",
    )
    add_gate(
        gates,
        "engine_stress.V6a_pass",
        stress.get("status") == "PASS"
        and all(item.get("status") == "PASS" for item in stress.get("validations", [])),
        "prior-corner accounting, positivity, and non-phantom stress checks",
        "technical",
    )
    data = forecast["data"]
    official = (
        data["DESI_DR1"]["labels"] == DR1.labels
        and data["DESI_DR2"]["labels"] == DR2.labels
        and np.array_equal(np.asarray(data["DESI_DR1"]["values"]), DR1.values)
        and np.array_equal(np.asarray(data["DESI_DR2"]["values"]), DR2.values)
        and np.array_equal(
            np.asarray(data["DESI_DR1"]["covariance"]), DR1.covariance
        )
        and np.array_equal(
            np.asarray(data["DESI_DR2"]["covariance"]), DR2.covariance
        )
    )
    add_gate(
        gates,
        "official_DESI_arrays",
        official,
        "embedded vectors and covariances equal the pinned likelihood",
        "technical",
    )
    summary = forecast["sample_summary"]
    add_gate(
        gates,
        "sample_accounting",
        summary["completed"] + summary["failed"] == summary["total"],
        f"completed={summary['completed']} failed={summary['failed']} "
        f"total={summary['total']}",
        "technical",
    )
    tables = forecast["dr3_forecast"]
    full_q = np.asarray(tables["full"]["quantiles"])
    add_gate(
        gates,
        "intervals_finite_and_ordered",
        bool(
            np.all(np.isfinite(full_q))
            and np.all(np.diff(full_q, axis=1) >= 0.0)
        ),
        "every DR3 quantile is finite and monotone in level",
        "technical",
    )

    # G1: sampling convergence under Sobol halving.
    g1_shift, g1_ratio = interval_comparison(tables["full"], tables["half_samples"])
    add_gate(
        gates,
        "G1.sampling_convergence",
        g1_shift < G1_MEDIAN_SHIFT_LIMIT and g1_ratio < G1_WIDTH_CHANGE_LIMIT,
        f"max median shift {g1_shift:.4f} (limit {G1_MEDIAN_SHIFT_LIMIT}); "
        f"max width change {g1_ratio:.4f} (limit {G1_WIDTH_CHANGE_LIMIT})",
        "scientific",
    )

    # G2: robustness to disjoint stochastic seed halves.
    g2_shift, _ = interval_comparison(tables["seed_split_a"], tables["seed_split_b"])
    add_gate(
        gates,
        "G2.seed_robustness",
        g2_shift < G2_MEDIAN_SHIFT_LIMIT,
        f"max seed-split median shift {g2_shift:.4f} "
        f"(limit {G2_MEDIAN_SHIFT_LIMIT})",
        "scientific",
    )

    # G3: predictive convergence and conditioning robustness.
    ess = forecast["effective_sample_size"]
    add_gate(
        gates,
        "G3.effective_sample_size",
        ess["joint"] >= G3_MINIMUM_ESS,
        f"joint ESS {ess['joint']:.1f} (minimum {G3_MINIMUM_ESS})",
        "scientific",
    )
    g3_shift, _ = interval_comparison(tables["full"], tables["dr2_only_weights"])
    add_gate(
        gates,
        "G3.dr2_only_sensitivity",
        g3_shift < G3_DR2_ONLY_SHIFT_LIMIT,
        f"DR2-only median shift {g3_shift:.4f} "
        f"(limit {G3_DR2_ONLY_SHIFT_LIMIT}); bounds the DR1/DR2 overlap caveat",
        "scientific",
    )
    width_ratio = np.asarray(tables["width_to_dr2_sigma_68"])
    add_gate(
        gates,
        "G3.informative_intervals",
        bool(np.all(width_ratio > 0.0) and np.all(np.isfinite(width_ratio))),
        f"68 widths span {float(np.min(width_ratio)):.2f}-"
        f"{float(np.max(width_ratio)):.2f} of DR2 sigma",
        "scientific",
    )

    # G4: end-to-end synthetic coverage calibration.
    coverage = calibration["gates"]
    add_gate(
        gates,
        "G4.coverage_calibration",
        calibration.get("status") == "PASS",
        f"95%: {coverage['coverage_95']['observed']:.3f} "
        f"(min {coverage['coverage_95']['required_minimum']}); "
        f"68%: {coverage['coverage_68']['observed']:.3f} "
        f"(range {coverage['coverage_68']['required_range']})",
        "scientific",
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
        status = "NO_FORECAST_V2"
    else:
        status = "FORECAST_ELIGIBLE"

    lcdm = np.asarray(forecast["baseline_overlays"]["LCDM"]["dr3_layout_prediction"])
    sigma = np.sqrt(np.diag(DR2.covariance))
    lcdm_offset = float(np.max(np.abs(full_q[:, 3] - lcdm) / sigma))
    return {
        "schema_version": "dscd-v2-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "question_answered": (
            "Do history-compatible DSCD realizations converge on a finite, "
            "stable, calibrated DR3 prediction?"
        ),
        "gate_counts": {
            "total": len(gates),
            "passed": sum(item["passed"] for item in gates),
            "technical_failures": len(technical_failures),
            "scientific_failures": len(scientific_failures),
        },
        "gates": gates,
        "summary": {
            "effective_sample_size_joint": ess["joint"],
            "sampling_convergence_max_shift": g1_shift,
            "seed_split_max_shift": g2_shift,
            "dr2_only_max_shift": g3_shift,
            "coverage_95": coverage["coverage_95"]["observed"],
            "coverage_68": coverage["coverage_68"]["observed"],
            "max_median_offset_from_lcdm_sigma": lcdm_offset,
        },
        "artifact_sha256": {
            name: file_sha256(path) for name, path in ARTIFACTS.items()
        },
        "interpretation": [
            "The forecast is an interval statement generated by forward evolution "
            "of history-compatible DSCD+GR realizations, not a parameter fit.",
            "A prediction coincident with LambdaCDM remains a falsifiable forecast; "
            "coincidence is a result, not a failure.",
            "Internal DSCD parameters remain individually degenerate; predictive "
            "convergence, not identifiability, is the v2 eligibility criterion.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    failed = [item for item in report["gates"] if not item["passed"]]
    summary = report["summary"]
    lines = [
        "# DSCD v2 forecast audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Question: {report['question_answered']}",
        "",
        "## Gate summary",
        "",
        f"- Passed: {report['gate_counts']['passed']} / {report['gate_counts']['total']}",
        f"- Technical failures: {report['gate_counts']['technical_failures']}",
        f"- Scientific failures: {report['gate_counts']['scientific_failures']}",
        "",
        "## Key numbers",
        "",
        f"- Joint effective sample size: {summary['effective_sample_size_joint']:.1f}",
        f"- G1 max median shift (half sampling): {summary['sampling_convergence_max_shift']:.4f}",
        f"- G2 max median shift (seed split): {summary['seed_split_max_shift']:.4f}",
        f"- DR2-only conditioning shift: {summary['dr2_only_max_shift']:.4f}",
        f"- Synthetic coverage: 95% level {summary['coverage_95']:.3f}, 68% level {summary['coverage_68']:.3f}",
        (
            "- Max forecast median offset from best-fit LambdaCDM: "
            f"{summary['max_median_offset_from_lcdm_sigma']:.3f} DR2 sigma"
        ),
        "",
    ]
    if failed:
        lines.extend(["## Failed gates", ""])
        lines.extend(f"- `{item['name']}`: {item['detail']}" for item in failed)
        lines.append("")
    lines.extend(
        ["## Interpretation", "", *[f"- {item}" for item in report["interpretation"]], ""]
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
