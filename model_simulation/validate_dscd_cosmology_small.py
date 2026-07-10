"""Run the mandatory V1-V5 small-validation ladder for DSCD cosmology."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar

from desi_analysis_common import file_sha256, write_json_atomic
from desi_bao_likelihood import BAODataset, Observable, bao_distances
from dscd_cosmology_config import DSCDCosmologyConfig, SYSTEM_VERSION
from dscd_cosmology_dynamics import (
    DSCDCosmologySystem,
    fund_count,
    interval_equation_of_state,
    pressure_from_depletion,
)
from dscd_cosmology_observables import trajectory_prediction
from dscd_cosmology_state import EventRequests


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "dscd_cosmology_results" / "small_validations.json"
SOURCE_FILES = (
    "DSCD_COSMOLOGY_SYSTEM_SPEC.md",
    "dscd_cosmology_config.py",
    "dscd_cosmology_state.py",
    "dscd_cosmology_dynamics.py",
    "dscd_cosmology_observables.py",
    "desi_bao_likelihood.py",
)


def _assert_close(
    observed: float | np.ndarray,
    expected: float | np.ndarray,
    tolerance: float,
    label: str,
) -> None:
    if not np.allclose(observed, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"{label}: observed={observed!r}, expected={expected!r}")


def _run_validation(
    name: str,
    expected: list[str],
    function: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    result = function()
    return {
        "name": name,
        "status": "PASS",
        "expected": expected,
        "observed": result,
        "runtime_seconds": time.perf_counter() - started,
    }


def validate_v1() -> dict[str, Any]:
    no_count, no_budget = fund_count(0, 3, 0.2, 1.0)
    exact_count, exact_budget = fund_count(5, 5, 0.2, 1.0)
    limited_count, limited_budget = fund_count(3, 3, 0.4, 0.5)
    if (no_count, exact_count, limited_count) != (0, 5, 1):
        raise AssertionError("funded counts do not match hand calculation")
    _assert_close([no_budget, exact_budget, limited_budget], [1.0, 0.0, 0.1], 1e-14, "budgets")

    config = DSCDCosmologyConfig(
        z_start=0.1,
        dN=0.01,
        n_cells=1,
        initial_rho_c=0.01,
        initial_x=3.0,
        initial_y=2.0,
        alpha_xy=0.0,
        alpha_xx=0.0,
        cost_xy=0.002,
        cost_xx=0.001,
        ripple_threshold=1.0e9,
        leakage_rate=0.0,
        explosion_rate=0.0,
        bond_rate=0.0,
        gamma_xy=0.2,
        gamma_xx=0.1,
        gamma_bond=0.0,
        diffusion_x=0.0,
        diffusion_y=0.0,
    ).validate()
    system = DSCDCosmologySystem(config)
    state = system.initial_state()
    request = EventRequests(
        xy=np.asarray([1]),
        xx=np.asarray([1]),
        explosion=np.asarray([0]),
        bonds=np.asarray([0]),
    )
    coefficient = config.omega_c_today / config.initial_rho_c
    next_state, diagnostics = system.step(
        state,
        0.001,
        coefficient,
        np.random.default_rng(1),
        request,
    )
    _assert_close(next_state.rho_c, [0.007], 1e-14, "deterministic density")
    _assert_close(next_state.x, [0.0], 1e-14, "deterministic X")
    _assert_close(next_state.y, [1.0], 1e-14, "deterministic Y")
    _assert_close(next_state.structure, [0.3], 1e-14, "deterministic structure")
    _assert_close(diagnostics.debit, [0.003], 1e-14, "exact debit")
    if diagnostics.w_interval < -1.0:
        raise AssertionError("positive depletion produced a phantom interval")
    return {
        "funding_cases": {
            "no_events": [no_count, no_budget],
            "exact_exhaustion": [exact_count, exact_budget],
            "insufficient_budget": [limited_count, limited_budget],
        },
        "deterministic_step": diagnostics.to_dict(),
        "next_state": {
            "rho_c": next_state.rho_c.tolist(),
            "x": next_state.x.tolist(),
            "y": next_state.y.tolist(),
            "structure": next_state.structure.tolist(),
        },
    }


def validate_v2() -> dict[str, Any]:
    config = DSCDCosmologyConfig(
        z_start=0.2,
        dN=0.05,
        n_cells=1,
        alpha_xy=0.5,
        alpha_xx=0.4,
        cost_xy=0.0,
        cost_xx=0.0,
        explosion_cost=0.0,
        bond_cost=0.0,
        ripple_threshold=1.0e9,
        leakage_rate=0.0,
        explosion_rate=0.0,
        bond_rate=0.0,
        diffusion_x=0.0,
        diffusion_y=0.0,
    ).validate()
    system = DSCDCosmologySystem(config)
    state = system.initial_state()
    coefficient = config.omega_c_today / config.initial_rho_c
    expansion = system.expansion(state, coefficient)
    dtau = config.dN / expansion
    xy_rate, xx_rate = system.interaction_rates(state)
    expected_xy = float(xy_rate[0] * dtau)
    expected_xx = float(xx_rate[0] * dtau)

    def draws(seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        xy: list[int] = []
        xx: list[int] = []
        for _ in range(2000):
            _, diagnostics = system.step(state, config.dN, coefficient, rng)
            xy.append(int(diagnostics.requested_xy[0]))
            xx.append(int(diagnostics.requested_xx[0]))
        return np.asarray(xy), np.asarray(xx)

    first_xy, first_xx = draws(9901)
    second_xy, second_xx = draws(9901)
    if not np.array_equal(first_xy, second_xy) or not np.array_equal(first_xx, second_xx):
        raise AssertionError("fixed stochastic seed is not reproducible")
    xy_tolerance = 5.0 * np.sqrt(max(expected_xy, 1e-12) / first_xy.size) + 0.01
    xx_tolerance = 5.0 * np.sqrt(max(expected_xx, 1e-12) / first_xx.size) + 0.01
    _assert_close(np.mean(first_xy), expected_xy, xy_tolerance, "XY Poisson mean")
    _assert_close(np.mean(first_xx), expected_xx, xx_tolerance, "XX Poisson mean")

    aligned = float(system.interference(np.asarray([0.0]))[0])
    opposed = float(system.interference(np.asarray([np.pi]))[0])
    if not aligned > opposed:
        raise AssertionError("beat phase does not modulate cross-interaction efficiency")

    no_beat = DSCDCosmologySystem(
        replace(config, omega_y=config.omega_x).validate()
    )
    no_beat_state = no_beat.initial_state()
    zero_requests = EventRequests.zeros(1)
    next_state, _ = no_beat.step(
        no_beat_state,
        config.dN,
        coefficient,
        np.random.default_rng(3),
        zero_requests,
    )
    _assert_close(next_state.phase, no_beat_state.phase, 1e-15, "zero-beat phase")

    no_memory = DSCDCosmologySystem(
        replace(config, gamma_xy=0.0, gamma_xx=0.0, gamma_bond=0.0).validate()
    )
    memory_state = no_memory.initial_state()
    memory_next, _ = no_memory.step(
        memory_state,
        config.dN,
        coefficient,
        np.random.default_rng(4),
        EventRequests(
            xy=np.asarray([1]),
            xx=np.asarray([0]),
            explosion=np.asarray([0]),
            bonds=np.asarray([0]),
        ),
    )
    _assert_close(memory_next.structure, memory_state.structure, 1e-15, "memory ablation")
    return {
        "expected_means": {"xy": expected_xy, "xx": expected_xx},
        "sample_means": {
            "xy": float(np.mean(first_xy)),
            "xx": float(np.mean(first_xx)),
        },
        "sample_variances": {
            "xy": float(np.var(first_xy, ddof=1)),
            "xx": float(np.var(first_xx, ddof=1)),
        },
        "interference": {"aligned": aligned, "opposed": opposed},
        "seed_reproducible": True,
        "zero_beat_phase_static": True,
        "memory_ablation_static": True,
    }


def validate_v3() -> dict[str, Any]:
    config = DSCDCosmologyConfig(
        z_start=0.2,
        dN=0.02,
        n_cells=3,
        initial_x=0.0,
        initial_y=0.0,
        alpha_xy=0.0,
        alpha_xx=0.0,
        leakage_rate=0.0,
        explosion_rate=0.0,
        bond_rate=0.0,
        diffusion_x=0.2,
        diffusion_y=0.15,
    ).validate()
    system = DSCDCosmologySystem(config)
    initial = np.asarray([3.0, 0.0, 0.0])
    full, residual = system._transport(initial, config.diffusion_x, 0.2)
    half, _ = system._transport(initial, config.diffusion_x, 0.1)
    half, _ = system._transport(half, config.diffusion_x, 0.1)
    _assert_close(np.sum(full), np.sum(initial), 1e-12, "transport conservation")
    if np.any(full < 0.0):
        raise AssertionError("transport produced a negative state")
    if np.linalg.norm(full - half) > 0.01:
        raise AssertionError("transport refinement changed the state excessively")

    absorbing = system.initial_state()
    if not system.is_absorbing(absorbing):
        raise AssertionError("empty zero-ripple state must be absorbing")
    separated = absorbing.copy()
    separated.x[:] = [1.0, 0.0, 0.0]
    separated.y[:] = [0.0, 1.0, 0.0]
    active_system = DSCDCosmologySystem(
        replace(config, alpha_xy=0.2).validate()
    )
    if active_system.is_absorbing(separated):
        raise AssertionError("transport-connectable separated species are not absorbing")
    return {
        "initial": initial.tolist(),
        "transported": full.tolist(),
        "two_half_steps": half.tolist(),
        "mass_residual": residual,
        "refinement_l2": float(np.linalg.norm(full - half)),
        "empty_absorbing": True,
        "separated_with_transport_absorbing": False,
    }


def validate_v4() -> dict[str, Any]:
    config = DSCDCosmologyConfig(
        z_start=1.0,
        dN=0.02,
        n_cells=1,
        initial_rho_c=1.0,
        initial_x=0.0,
        initial_y=0.0,
        alpha_xy=0.0,
        alpha_xx=0.0,
        leakage_rate=0.0,
        explosion_rate=0.0,
        bond_rate=0.0,
        diffusion_x=0.0,
        diffusion_y=0.0,
    ).validate()
    trajectory = DSCDCosmologySystem(config).simulate(31)
    expected = np.sqrt(
        config.omega_m * np.exp(-3.0 * trajectory.N)
        + config.omega_r * np.exp(-4.0 * trajectory.N)
        + config.omega_c_today
    )
    _assert_close(trajectory.expansion, expected, 2e-7, "D=0 Lambda background")

    rho = 1.2
    depletion = 0.08
    expansion = 1.1
    pressure = pressure_from_depletion(rho, depletion, expansion)
    residual = -depletion + 3.0 * expansion * (rho + pressure)
    _assert_close(residual, 0.0, 1e-14, "forced-depletion continuity")
    after = rho - depletion * 0.01
    w_interval = interval_equation_of_state(rho, after, 0.011)
    if w_interval < -1.0:
        raise AssertionError("constant positive depletion produced phantom w")

    active_config = replace(
        config,
        initial_x=6.0,
        initial_y=7.0,
        alpha_xy=0.8,
        alpha_xx=0.2,
        cost_xy=8.0e-4,
        cost_xx=4.0e-4,
        gamma_xy=5.0e-4,
        gamma_xx=2.0e-4,
    ).validate()
    active = DSCDCosmologySystem(active_config).simulate(31)
    if not active.rho_c[-1] < active.rho_c[0]:
        raise AssertionError("active DSCD trajectory did not deplete")
    if np.max(np.abs(active.continuity_residual)) > 1e-10:
        raise AssertionError("active continuity residual exceeded tolerance")
    if np.max(np.abs(active.friedmann_residual)) > 1e-10:
        raise AssertionError("active Friedmann residual exceeded tolerance")
    return {
        "lambda_max_abs_error": float(np.max(np.abs(trajectory.expansion - expected))),
        "forced_depletion_continuity_residual": float(residual),
        "forced_depletion_w_interval": float(w_interval),
        "active_rho_initial": float(active.rho_c[0]),
        "active_rho_final": float(active.rho_c[-1]),
        "active_max_continuity_residual": float(
            np.max(np.abs(active.continuity_residual))
        ),
        "active_max_friedmann_residual": float(
            np.max(np.abs(active.friedmann_residual))
        ),
    }


def validate_v5() -> dict[str, Any]:
    active_config = DSCDCosmologyConfig(
        z_start=1.4,
        dN=0.015,
        n_cells=2,
        initial_x=8.0,
        initial_y=9.0,
        alpha_xy=0.8,
        alpha_xx=0.2,
        cost_xy=0.0015,
        cost_xx=0.0008,
        gamma_xy=6.0e-4,
        gamma_xx=3.0e-4,
        diffusion_x=0.02,
        diffusion_y=0.02,
    ).validate()
    trajectory = DSCDCosmologySystem(active_config).simulate(77)
    toy = BAODataset(
        "synthetic_DSCD",
        (
            Observable(0.30, "DV", "toy-low"),
            Observable(0.75, "DM", "toy-mid"),
            Observable(0.75, "DH", "toy-mid"),
            Observable(1.20, "DV", "toy-high"),
        ),
        np.zeros(4),
        np.eye(4) * 0.01,
        {"synthetic": True},
    )
    truth_theta = 0.0334
    values = trajectory_prediction(trajectory, toy, truth_theta, 64)
    synthetic = BAODataset(
        toy.name,
        toy.observables,
        values,
        toy.covariance,
        toy.source,
    )

    objective = lambda theta: synthetic.chi2(
        trajectory_prediction(trajectory, synthetic, float(theta), 64)
    )
    fit = minimize_scalar(
        objective,
        bounds=(0.02, 0.045),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    _assert_close(fit.x, truth_theta, 2e-8, "synthetic theta recovery")

    no_dynamics = DSCDCosmologySystem(
        replace(
            active_config,
            initial_x=0.0,
            initial_y=0.0,
            alpha_xy=0.0,
            alpha_xx=0.0,
            leakage_rate=0.0,
            explosion_rate=0.0,
            bond_rate=0.0,
        ).validate()
    ).simulate(77)
    wrong = trajectory_prediction(no_dynamics, synthetic, truth_theta, 64)
    wrong_difference = float(np.linalg.norm(wrong - values))
    if wrong_difference <= 1.0e-6:
        raise AssertionError("no-dynamics ablation is numerically indistinguishable")

    # Independent LCDM utility check: it is a control only.
    lcdm_dm, lcdm_dh, lcdm_dv = bao_distances(
        np.asarray([0.30]),
        truth_theta,
        active_config.omega_m,
        lambda z: np.ones_like(z),
        omega_r=active_config.omega_r,
        quadrature_order=64,
    )
    return {
        "truth_theta": truth_theta,
        "recovered_theta": float(fit.x),
        "minimum_chi2": float(fit.fun),
        "synthetic_values": values.tolist(),
        "no_dynamics_l2_difference": wrong_difference,
        "lcdm_control_at_z_0p3": {
            "DM": float(lcdm_dm[0]),
            "DH": float(lcdm_dh[0]),
            "DV": float(lcdm_dv[0]),
        },
    }


def validate_v6a() -> dict[str, Any]:
    """V6a: engine stress at the extreme corners of the v2 forecast prior.

    Version 2 samples depletion scales up to one thousand times the v1
    values, so the engine must be verified there: funded accounting stays
    exact, density never goes negative, depletion is pathwise monotone and
    non-phantom, and any physically impossible realization fails with a
    clean typed exception instead of a corrupted state.
    """
    from dscd_forecast_prior import PriorSample, PRIOR_LOWER, PRIOR_UPPER, sample_prior

    def audited_run(config: DSCDCosmologyConfig, seed: int) -> dict[str, Any]:
        system = DSCDCosmologySystem(config)
        state = system.initial_state()
        coefficient = config.omega_c_today / config.initial_rho_c
        rng = np.random.default_rng(seed)
        steps = int(np.ceil(-config.n_start / config.dN))
        max_accounting_error = 0.0
        min_w_interval = np.inf
        for _ in range(steps):
            rho_before = state.rho_c.copy()
            state, diagnostics = system.step(state, config.dN, coefficient, rng)
            expected_debit = (
                diagnostics.funded_xy * config.cost_xy
                + diagnostics.funded_xx * config.cost_xx
                + diagnostics.funded_explosion * config.explosion_cost
                + diagnostics.funded_bonds * config.bond_cost
                + diagnostics.leakage
            )
            max_accounting_error = max(
                max_accounting_error,
                float(np.max(np.abs(diagnostics.debit - expected_debit))),
            )
            if np.any(state.rho_c < 0.0):
                raise AssertionError("stress run produced a negative density")
            if np.any(state.rho_c - rho_before > 1.0e-13):
                raise AssertionError("stress run increased the DSCD density")
            min_w_interval = min(min_w_interval, float(diagnostics.w_interval))
        if max_accounting_error > 1.0e-12:
            raise AssertionError(
                f"funded accounting drift {max_accounting_error:.3e} exceeds 1e-12"
            )
        if min_w_interval < -1.0 - 1.0e-12:
            raise AssertionError("funded depletion produced a phantom interval")
        return {
            "max_accounting_error": max_accounting_error,
            "min_w_interval": min_w_interval,
            "final_min_rho": float(np.min(state.rho_c)),
        }

    corner = PriorSample(
        0,
        np.asarray(
            [PRIOR_UPPER[0], PRIOR_UPPER[1], PRIOR_UPPER[2], PRIOR_UPPER[3],
             PRIOR_UPPER[4], PRIOR_UPPER[5], PRIOR_UPPER[6], 0.30]
        ),
    )
    corner_report = audited_run(corner.to_config(), 424242)

    mild = PriorSample(
        1,
        np.asarray(
            [PRIOR_LOWER[0], PRIOR_LOWER[1], PRIOR_LOWER[2], PRIOR_LOWER[3],
             PRIOR_LOWER[4], PRIOR_LOWER[5], PRIOR_LOWER[6], 0.30]
        ),
    )
    mild_report = audited_run(mild.to_config(), 424243)

    allowed = (ValueError, ArithmeticError, OverflowError)
    sweep_ok = 0
    sweep_failed_cleanly = 0
    failure_types: dict[str, int] = {}
    for sample in sample_prior(32, 626_262):
        try:
            trajectory = DSCDCosmologySystem(sample.to_config()).simulate(
                900_000 + sample.index
            )
            if np.any(trajectory.rho_c <= 0.0):
                raise AssertionError("completed trajectory reported non-positive density")
            if np.any(np.diff(trajectory.rho_c) > 1.0e-13):
                raise AssertionError("completed trajectory increased density")
            sweep_ok += 1
        except allowed as exc:
            sweep_failed_cleanly += 1
            key = type(exc).__name__
            failure_types[key] = failure_types.get(key, 0) + 1
    if sweep_ok + sweep_failed_cleanly != 32:
        raise AssertionError("prior sweep lost a sample")
    if sweep_ok == 0:
        raise AssertionError("no prior sample completed; the prior support is empty")
    return {
        "extreme_corner": corner_report,
        "mild_corner": mild_report,
        "prior_sweep": {
            "samples": 32,
            "completed": sweep_ok,
            "clean_typed_failures": sweep_failed_cleanly,
            "failure_types": failure_types,
        },
    }


def run_v6a() -> dict[str, Any]:
    validation = _run_validation(
        "V6a_prior_engine_stress",
        [
            "Funded accounting is exact at thousandfold depletion scales.",
            "Density is pathwise non-negative and non-increasing everywhere on the prior.",
            "Depletion never produces a phantom interval.",
            "Impossible realizations fail with clean typed exceptions.",
        ],
        validate_v6a,
    )
    return {
        "schema_version": "dscd-v2-engine-stress-v1",
        "system_version": SYSTEM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_sha256": {
            name: file_sha256(HERE / name)
            for name in SOURCE_FILES + ("dscd_forecast_prior.py",)
        },
        "validations": [validation],
        "real_desi_used": False,
        "next_gate": "history_ensemble",
    }


def run() -> dict[str, Any]:
    validations = [
        _run_validation(
            "V1_deterministic_cell",
            [
                "Every event is funded before commitment.",
                "Exact debit equals the sum of funded costs.",
                "The manually calculated state is reproduced.",
            ],
            validate_v1,
        ),
        _run_validation(
            "V2_stochastic_core",
            [
                "Requested event moments match declared Poisson hazards.",
                "Fixed seeds are exactly reproducible.",
                "Beat and memory ablations affect their intended mechanisms.",
            ],
            validate_v2,
        ),
        _run_validation(
            "V3_transport_graph",
            [
                "Undirected transport conserves activity and remains non-negative.",
                "Refining the transport step gives a nearby state.",
                "Separated species connected by diffusion are not called absorbing.",
            ],
            validate_v3,
        ),
        _run_validation(
            "V4_minimal_GR",
            [
                "Zero depletion reproduces flat LambdaCDM.",
                "The pressure closure satisfies continuity.",
                "Active DSCD evolution satisfies Friedmann and continuity constraints.",
            ],
            validate_v4,
        ),
        _run_validation(
            "V5_synthetic_distances",
            [
                "Distances are generated from an emergent trajectory.",
                "The exposed synthetic theta is recovered.",
                "A no-dynamics ablation does not reproduce the same trajectory.",
            ],
            validate_v5,
        ),
    ]
    return {
        "schema_version": "dscd-small-validation-v1",
        "system_version": SYSTEM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "order_enforced": [item["name"] for item in validations],
        "source_sha256": {
            name: file_sha256(HERE / name) for name in SOURCE_FILES
        },
        "validations": validations,
        "real_desi_used": False,
        "next_gate": "pilot_ensemble",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("v1_v5", "v6a"),
        default="v1_v5",
        help="v1_v5 preserves the historical ladder; v6a is the v2 prior stress gate.",
    )
    arguments = parser.parse_args()
    if arguments.stage == "v6a":
        report = run_v6a()
        default_output = HERE / "dscd_v2_results" / "engine_stress.json"
    else:
        report = run()
        default_output = DEFAULT_OUTPUT
    destination = write_json_atomic(
        report, arguments.output or default_output, arguments.force
    )
    print(f"Wrote {destination}")
