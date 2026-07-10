"""Write the mandatory DSCD DR3 forecast/no-forecast disposition."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from desi_analysis_common import file_sha256, write_json_atomic


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_DIR = HERE / "dscd_cosmology_results"
AUDIT_PATH = RESULT_DIR / "audit.json"
HISTORICAL_RECORD = ROOT / "16_CASCADE_ORIGIN_AND_DR3_PREDICTION_RECORD.md"
DEFAULT_JSON = RESULT_DIR / "forecast_disposition.json"
DEFAULT_MARKDOWN = ROOT / "17_CORRECTED_DSCD_DR3_DISPOSITION.md"


def load_audit() -> dict[str, Any]:
    with AUDIT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_disposition() -> dict[str, Any]:
    audit = load_audit()
    if audit.get("status") not in {"FORECAST_ELIGIBLE", "NO_FORECAST"}:
        raise RuntimeError("scientific audit has no valid forecast disposition")
    failed = [item for item in audit["gates"] if not item["passed"]]
    status = audit["status"]
    return {
        "schema_version": "dscd-forecast-disposition-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "audit_sha256": file_sha256(AUDIT_PATH),
        "historical_record": str(HISTORICAL_RECORD.name),
        "historical_record_sha256": file_sha256(HISTORICAL_RECORD),
        "historical_record_preserved": True,
        "new_DR3_numbers_issued": status == "FORECAST_ELIGIBLE",
        "decision": (
            "The corrected coupled DSCD+GR system is not eligible to issue a "
            "new DR3 forecast. No corrected prediction numbers are sealed."
            if status == "NO_FORECAST"
            else "The corrected system passed every gate and may issue a separate forecast."
        ),
        "failed_gates": failed,
        "audit_summary": audit["summary"],
        "rules": [
            "This disposition does not alter or erase the historical prediction record.",
            "The historical constant-w surrogate is not the corrected DSCD+GR system.",
            "A later forecast requires a new versioned system and a complete new pre-release audit.",
            "DR3 evaluation requires a disjoint increment or cross-release covariance for independence.",
        ],
    }


def write_markdown(payload: dict[str, Any], path: Path, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force")
    lines = [
        "# Corrected cosmological DSCD forecast disposition",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Created (UTC): `{payload['created_utc']}`",
        f"Scientific audit SHA-256: `{payload['audit_sha256']}`",
        (
            "Historical record SHA-256: "
            f"`{payload['historical_record_sha256']}`"
        ),
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "The historical file `16_CASCADE_ORIGIN_AND_DR3_PREDICTION_RECORD.md` "
        "was preserved byte-for-byte. Its constant-w surrogate is historical "
        "provenance and is not a prediction from the corrected coupled DSCD+GR system.",
        "",
    ]
    if payload["failed_gates"]:
        lines.extend(["## Blocking scientific gates", ""])
        lines.extend(
            f"- `{item['name']}`: {item['detail']}"
            for item in payload["failed_gates"]
        )
        lines.append("")
    lines.extend(
        [
            "## Amendment rule",
            "",
            "No corrected DR3 numerical forecast may be added to this disposition. "
            "A materially revised dynamical system requires a new versioned record, "
            "new source/configuration hashes, and a complete pre-release audit.",
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
    before_hash = file_sha256(HISTORICAL_RECORD)
    disposition = build_disposition()
    json_path = write_json_atomic(disposition, arguments.json, arguments.force)
    markdown_path = write_markdown(disposition, arguments.markdown, arguments.force)
    after_hash = file_sha256(HISTORICAL_RECORD)
    if before_hash != after_hash:
        raise RuntimeError("historical prediction record changed during disposition write")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path} ({disposition['status']})")
