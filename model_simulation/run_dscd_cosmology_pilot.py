"""Run a modest no-DESI pilot ensemble after the V1-V5 gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desi_analysis_common import file_sha256, write_json_atomic
from dscd_cosmology_config import DSCDCosmologyConfig, SYSTEM_VERSION
from dscd_cosmology_ensemble import DSCDEnsembleResult, run_ensemble
from dscd_cosmology_observables import (
    acceleration_parameter,
    early_fraction_bound,
    trajectory_bao_distances,
)


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "dscd_cosmology_results"
SMALL_VALIDATION = RESULT_DIR / "small_validations.json"
DEFAULT_OUTPUT = RESULT_DIR / "pilot_ensemble.json"
PILOT_SEEDS = (101, 103, 107, 109)
Z_GRID = np.asarray([0.0, 0.3, 0.6, 0.9, 1.2, 1.8, 2.3])


def require_small_validation() -> dict[str, Any]:
    with SMALL_VALIDATION.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "PASS" or report.get("real_desi_used") is not False:
        raise RuntimeError("V1-V5 validation artifact is absent or invalid")
    required = {
        "V1_deterministic_cell",
        "V2_stochastic_core",
        "V3_transport_graph",
        "V4_minimal_GR",
        "V5_synthetic_distances",
    }
    observed = {item["name"] for item in report.get("validations", [])}
    if observed != required or any(
        item.get("status") != "PASS" for item in report["validations"]
    ):
        raise RuntimeError("not every V1-V5 stage passed")
    return report


def expansion_samples(
    ensemble: DSCDEnsembleResult, redshifts: np.ndarray = Z_GRID
) -> np.ndarray:
    return np.stack(
        [trajectory.expansion_at(redshifts) for trajectory in ensemble.trajectories]
    )


def summarize_ensemble(ensemble: DSCDEnsembleResult) -> dict[str, Any]:
    samples = expansion_samples(ensemble)
    final_density = np.asarray(
        [trajectory.rho_c[-1] for trajectory in ensemble.trajectories]
    )
    minimum_w = np.asarray(
        [np.min(trajectory.w_interval) for trajectory in ensemble.trajectories]
    )
    maximum_w = np.asarray(
        [np.max(trajectory.w_interval) for trajectory in ensemble.trajectories]
    )
    return {
        "seeds": list(ensemble.seeds),
        "z_grid": Z_GRID.tolist(),
        "expansion_median": np.median(samples, axis=0).tolist(),
        "expansion_p16": np.percentile(samples, 16.0, axis=0).tolist(),
        "expansion_p84": np.percentile(samples, 84.0, axis=0).tolist(),
        "final_density": final_density.tolist(),
        "minimum_w_interval": minimum_w.tolist(),
        "maximum_w_interval": maximum_w.tolist(),
        "all_nonphantom": bool(np.all(minimum_w >= -1.0 - 1.0e-12)),
        "all_closure_converged": all(
            trajectory.closure_converged for trajectory in ensemble.trajectories
        ),
        "max_continuity_residual": float(
            max(
                np.max(np.abs(trajectory.continuity_residual))
                for trajectory in ensemble.trajectories
            )
        ),
        "max_friedmann_residual": float(
            max(
                np.max(np.abs(trajectory.friedmann_residual))
                for trajectory in ensemble.trajectories
            )
        ),
    }


def run() -> dict[str, Any]:
    validation = require_small_validation()
    config = DSCDCosmologyConfig().validate()
    base = run_ensemble(config, PILOT_SEEDS, variance_semantics="physical")
    base_samples = expansion_samples(base)
    base_median = np.median(base_samples, axis=0)

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
    for name, ablation_config in ablation_configs.items():
        ensemble = run_ensemble(
            ablation_config, PILOT_SEEDS, variance_semantics="physical"
        )
        samples = expansion_samples(ensemble)
        median = np.median(samples, axis=0)
        ablations[name] = {
            "summary": summarize_ensemble(ensemble),
            "max_abs_expansion_delta_from_complete": float(
                np.max(np.abs(median - base_median))
            ),
        }

    resolution: dict[str, Any] = {}
    reference_expansion: np.ndarray | None = None
    for d_n in (0.04, 0.02, 0.01):
        trajectory = run_ensemble(
            replace(config, dN=d_n).validate(),
            [PILOT_SEEDS[0]],
            variance_semantics="monte_carlo",
        ).trajectories[0]
        values = trajectory.expansion_at(Z_GRID)
        resolution[str(d_n)] = {
            "steps": int(trajectory.N.size - 1),
            "expansion": values.tolist(),
        }
        if d_n == 0.01:
            reference_expansion = values
    assert reference_expansion is not None
    for item in resolution.values():
        item["max_abs_delta_from_dN_0p01"] = float(
            np.max(np.abs(np.asarray(item["expansion"]) - reference_expansion))
        )

    graph_refinement: dict[str, Any] = {}
    for cells in (4, 8, 16):
        ensemble = run_ensemble(
            replace(config, n_cells=cells).validate(),
            PILOT_SEEDS[:2],
            variance_semantics="monte_carlo",
        )
        graph_refinement[str(cells)] = summarize_ensemble(ensemble)

    representative = base.trajectories[0]
    theta_fixture = 0.033
    dm, dh, dv = trajectory_bao_distances(
        representative, Z_GRID, theta_fixture, quadrature_order=64
    )
    midpoint_n, deceleration = acceleration_parameter(representative)
    early_fraction = early_fraction_bound(
        representative, config.omega_m, config.omega_r, redshift=1100.0
    )

    hard_checks = {
        "small_validation_passed": validation["status"] == "PASS",
        "all_closure_converged": all(
            trajectory.closure_converged for trajectory in base.trajectories
        ),
        "all_density_positive": all(
            np.all(trajectory.rho_c > 0.0) for trajectory in base.trajectories
        ),
        "all_density_nonincreasing": all(
            np.all(np.diff(trajectory.rho_c) <= 1.0e-13)
            for trajectory in base.trajectories
        ),
        "all_interval_w_nonphantom": all(
            np.all(trajectory.w_interval >= -1.0 - 1.0e-12)
            for trajectory in base.trajectories
        ),
        "continuity_residual_below_1e-10": max(
            np.max(np.abs(trajectory.continuity_residual))
            for trajectory in base.trajectories
        )
        < 1.0e-10,
        "friedmann_residual_below_1e-10": max(
            np.max(np.abs(trajectory.friedmann_residual))
            for trajectory in base.trajectories
        )
        < 1.0e-10,
        "early_fraction_below_1e-8": early_fraction < 1.0e-8,
    }
    status = "PASS" if all(hard_checks.values()) else "FAIL"
    return {
        "schema_version": "dscd-pilot-v1",
        "system_version": SYSTEM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "real_desi_used": False,
        "small_validation_sha256": file_sha256(SMALL_VALIDATION),
        "source_sha256": {
            name: file_sha256(HERE / name)
            for name in (
                "DSCD_COSMOLOGY_SYSTEM_SPEC.md",
                "dscd_cosmology_config.py",
                "dscd_cosmology_state.py",
                "dscd_cosmology_dynamics.py",
                "dscd_cosmology_ensemble.py",
                "dscd_cosmology_observables.py",
                "run_dscd_cosmology_pilot.py",
            )
        },
        "config": config.to_dict(),
        "base": summarize_ensemble(base),
        "ablations": ablations,
        "resolution": resolution,
        "graph_refinement": graph_refinement,
        "representative_trajectory": representative.to_dict(include_series=True),
        "fixture_distances": {
            "theta": theta_fixture,
            "z": Z_GRID.tolist(),
            "DM_over_rd": dm.tolist(),
            "DH_over_rd": dh.tolist(),
            "DV_over_rd": dv.tolist(),
        },
        "acceleration": {
            "N_midpoint": midpoint_n.tolist(),
            "q": deceleration.tolist(),
            "accelerating_today": bool(deceleration[-1] < 0.0),
        },
        "early_fraction_z1100": early_fraction,
        "hard_checks": hard_checks,
        "next_gate": "identifiability_and_convergence",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    report = run()
    destination = write_json_atomic(report, arguments.output, arguments.force)
    print(f"Wrote {destination} ({report['status']})")
