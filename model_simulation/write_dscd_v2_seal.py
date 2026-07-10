"""Write the versioned v2 DR3 forecast seal, if and only if the audit passes.

The seal is a falsifiable interval statement produced by forward evolution
of history-compatible DSCD+GR realizations.  The historical record
`16_CASCADE_ORIGIN_AND_DR3_PREDICTION_RECORD.md` and every v1 artifact are
preserved byte-for-byte; this file writes only the new, versioned record.
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
from desi_bao_likelihood import DR2, PINNED_COMMIT


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_DIR = HERE / "dscd_v2_results"
AUDIT_PATH = RESULT_DIR / "audit_v2.json"
FORECAST_PATH = RESULT_DIR / "forecast_production.json"
CALIBRATION_PATH = RESULT_DIR / "calibration.json"
SEAL_JSON = RESULT_DIR / "dr3_seal_v2.json"
SEAL_MARKDOWN = ROOT / "18_DSCD_V2_DR3_FORECAST_RECORD.md"


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_seal() -> dict[str, Any]:
    audit = load(AUDIT_PATH)
    if audit.get("status") != "FORECAST_ELIGIBLE":
        raise RuntimeError(
            f"audit status is {audit.get('status')!r}; no seal may be written"
        )
    forecast = load(FORECAST_PATH)
    calibration = load(CALIBRATION_PATH)
    table = forecast["dr3_forecast"]["full"]
    quantiles = np.asarray(table["quantiles"])
    levels = table["quantile_levels"]

    def column(level: float) -> np.ndarray:
        return quantiles[:, levels.index(level)]

    observables = []
    for row, label in enumerate(table["labels"]):
        observables.append(
            {
                "label": label,
                "median": float(column(0.50)[row]),
                "interval_68": [float(column(0.16)[row]), float(column(0.84)[row])],
                "interval_95": [float(column(0.025)[row]), float(column(0.975)[row])],
                "interval_99": [float(column(0.005)[row]), float(column(0.995)[row])],
                "dr2_sigma": float(np.sqrt(DR2.covariance[row, row])),
            }
        )
    return {
        "schema_version": "dscd-v2-dr3-seal-v1",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_version": forecast["forecast_version"],
        "system_version": forecast["system_version"],
        "audit_status": audit["status"],
        "audit_sha256": file_sha256(AUDIT_PATH),
        "forecast_sha256": file_sha256(FORECAST_PATH),
        "calibration_sha256": file_sha256(CALIBRATION_PATH),
        "official_data_commit": PINNED_COMMIT,
        "prior": forecast["prior"],
        "sampler_seed": forecast["sampler_seed"],
        "settings": forecast["settings"],
        "effective_sample_size": forecast["effective_sample_size"]["joint"],
        "coverage": {
            "level_95": calibration["gates"]["coverage_95"]["observed"],
            "level_68": calibration["gates"]["coverage_68"]["observed"],
        },
        "caveats": forecast["caveats"],
        "observables": observables,
        "falsification_criteria": [
            "Primary: any official DESI DR3 measured central value for a sealed "
            "observable lying outside the sealed 99% interval widened by one DR3 "
            "measurement sigma on each side falsifies the v2 forecast.",
            "Structural: funded DSCD depletion cannot produce phantom behavior; "
            "a robust DR3-era background reconstruction requiring w < -1 across "
            "the sealed redshift range falsifies the DSCD dark-sector mechanism.",
            "Meta: any post-DR3 modification of prior bounds, sampler seed, or "
            "gate thresholds voids this seal; a corrected forecast must be "
            "re-sealed under a new version.",
        ],
    }


def write_markdown(seal: dict[str, Any], path: Path, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    rows = [
        "| Observable | Median | 68% interval | 95% interval | 99% interval |",
        "|---|---|---|---|---|",
    ]
    for item in seal["observables"]:
        rows.append(
            f"| `{item['label']}` | {item['median']:.3f} "
            f"| [{item['interval_68'][0]:.3f}, {item['interval_68'][1]:.3f}] "
            f"| [{item['interval_95'][0]:.3f}, {item['interval_95'][1]:.3f}] "
            f"| [{item['interval_99'][0]:.3f}, {item['interval_99'][1]:.3f}] |"
        )
    lines = [
        "# DSCD v2 DR3 forecast record (sealed)",
        "",
        f"- Sealed (UTC): {seal['sealed_utc']}",
        f"- Forecast version: `{seal['forecast_version']}` on `{seal['system_version']}`",
        f"- Audit status at sealing: `{seal['audit_status']}`",
        f"- Audit artifact sha256: `{seal['audit_sha256']}`",
        f"- Forecast artifact sha256: `{seal['forecast_sha256']}`",
        f"- Calibration artifact sha256: `{seal['calibration_sha256']}`",
        f"- Official BAO data commit: `{seal['official_data_commit']}`",
        f"- Joint effective sample size: {seal['effective_sample_size']:.1f}",
        (
            "- Synthetic coverage at sealing: "
            f"95% level {seal['coverage']['level_95']:.3f}, "
            f"68% level {seal['coverage']['level_68']:.3f}"
        ),
        "",
        "## What is sealed",
        "",
        "Forward predictions for the DESI DR3 tracer layout (the DR2 tracers "
        "remeasured), generated by evolving history-compatible Discrete "
        "Stochastic Cascade Dynamics realizations coupled to flat FLRW general "
        "relativity.  Realizations were weighted by their joint DR1+DR2 "
        "likelihood; nothing was fitted to any unreleased data.  Distances are "
        "dimensionless (units of the drag sound horizon).",
        "",
        rows[0],
        rows[1],
        *rows[2:],
        "",
        "## Falsification criteria",
        "",
        *[f"{index + 1}. {item}" for index, item in enumerate(seal["falsification_criteria"])],
        "",
        "## Declared caveats",
        "",
        *[f"- {item}" for item in seal["caveats"]],
        "",
        "## Relation to historical records",
        "",
        "- `16_CASCADE_ORIGIN_AND_DR3_PREDICTION_RECORD.md` is a historical "
        "document preserved unchanged; its numbers came from a superseded "
        "surrogate analysis and are not part of this seal.",
        "- `17_CORRECTED_DSCD_DR3_DISPOSITION.md` and the v1 coupled-system "
        "audit (`model_simulation/dscd_cosmology_results/`) returned "
        "NO_FORECAST for one frozen configuration; version 2 replaces that "
        "frozen configuration with a declared prior over latent states and "
        "gates on predictive convergence.",
        "",
    ]
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
    parser.add_argument("--json", type=Path, default=SEAL_JSON)
    parser.add_argument("--markdown", type=Path, default=SEAL_MARKDOWN)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    seal = build_seal()
    json_path = write_json_atomic(seal, arguments.json, arguments.force)
    markdown_path = write_markdown(seal, arguments.markdown, arguments.force)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
