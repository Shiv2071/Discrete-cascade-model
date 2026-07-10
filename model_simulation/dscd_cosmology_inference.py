"""Inference helpers for trajectory-generated DSCD cosmology observables."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from desi_bao_likelihood import BAODataset
from dscd_cosmology_config import DSCDCosmologyConfig
from dscd_cosmology_ensemble import run_ensemble
from dscd_cosmology_observables import trajectory_prediction


PARAMETER_NAMES = ("theta", "omega_m", "depletion_scale")
PARAMETER_BOUNDS = ((0.020, 0.045), (0.10, 0.60), (0.0, 4.0))


def apply_depletion_scale(
    config: DSCDCosmologyConfig, depletion_scale: float
) -> DSCDCosmologyConfig:
    """Scale one declared DSCD density-depletion combination.

    This is the only DSCD combination exposed to BAO inference in version 1.
    It scales every funded density cost while preserving the internal event,
    beat, memory, regime, and transport dynamics.
    """
    if not np.isfinite(depletion_scale) or depletion_scale < 0.0:
        raise ValueError("depletion_scale must be finite and non-negative")
    return replace(
        config,
        cost_xy=config.cost_xy * depletion_scale,
        cost_xx=config.cost_xx * depletion_scale,
        explosion_cost=config.explosion_cost * depletion_scale,
        bond_cost=config.bond_cost * depletion_scale,
        leakage_rate=config.leakage_rate * depletion_scale,
    ).validate()


@dataclass(frozen=True)
class DSCDFitResult:
    parameters: np.ndarray
    objective: float
    chi2_observational: float
    joint_log_predictive_density: float
    prediction_mean: np.ndarray
    prediction_covariance: np.ndarray
    active_bounds: tuple[str, ...]
    global_runs: tuple[dict[str, Any], ...]
    local_message: str
    fit_depletion: bool
    optimization_seeds: tuple[int, ...]
    trajectory_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        names = PARAMETER_NAMES if self.fit_depletion else PARAMETER_NAMES[:2]
        return {
            "parameter_names": list(names),
            "parameters": {
                name: float(value) for name, value in zip(names, self.parameters)
            },
            "parameter_vector": self.parameters.tolist(),
            "objective": float(self.objective),
            "chi2_observational": float(self.chi2_observational),
            "joint_log_predictive_density": float(
                self.joint_log_predictive_density
            ),
            "prediction_mean": self.prediction_mean.tolist(),
            "prediction_covariance": self.prediction_covariance.tolist(),
            "active_bounds": list(self.active_bounds),
            "global_runs": list(self.global_runs),
            "local_message": self.local_message,
            "fit_depletion": bool(self.fit_depletion),
            "optimization_seeds": list(self.optimization_seeds),
            "trajectory_seeds": list(self.trajectory_seeds),
        }


class DSCDTrajectoryLikelihood:
    """Full-covariance likelihood driven by complete stochastic trajectories."""

    def __init__(
        self,
        base_config: DSCDCosmologyConfig,
        dataset: BAODataset,
        trajectory_seeds: Sequence[int],
        *,
        quadrature_order: int = 48,
        physical_variance: bool = True,
    ) -> None:
        self.base_config = base_config.validate()
        self.dataset = dataset
        self.trajectory_seeds = tuple(int(seed) for seed in trajectory_seeds)
        if not self.trajectory_seeds:
            raise ValueError("at least one trajectory seed is required")
        self.quadrature_order = int(quadrature_order)
        self.physical_variance = bool(physical_variance)
        self._cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}

    def predictive_distribution(
        self, parameters: Sequence[float], fit_depletion: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(parameters, dtype=float)
        expected_size = 3 if fit_depletion else 2
        if p.shape != (expected_size,) or np.any(~np.isfinite(p)):
            raise ValueError("invalid DSCD parameter vector")
        theta, omega_m = float(p[0]), float(p[1])
        depletion_scale = float(p[2]) if fit_depletion else 1.0
        bounds = PARAMETER_BOUNDS[:expected_size]
        for value, (lower, upper) in zip(p, bounds):
            if value < lower or value > upper:
                raise ValueError("DSCD parameter lies outside its declared bound")

        key = tuple(float(value) for value in p) + (float(fit_depletion),)
        if key in self._cache:
            mean, covariance = self._cache[key]
            return mean.copy(), covariance.copy()

        config = apply_depletion_scale(
            replace(self.base_config, omega_m=omega_m).validate(),
            depletion_scale,
        )
        ensemble = run_ensemble(
            config,
            self.trajectory_seeds,
            variance_semantics="physical" if self.physical_variance else "monte_carlo",
        )
        samples = np.stack(
            [
                trajectory_prediction(
                    trajectory,
                    self.dataset,
                    theta,
                    quadrature_order=self.quadrature_order,
                )
                for trajectory in ensemble.trajectories
            ]
        )
        mean = np.mean(samples, axis=0)
        if self.physical_variance and samples.shape[0] > 1:
            covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
        else:
            covariance = np.zeros_like(self.dataset.covariance)
        self._cache[key] = (mean.copy(), covariance.copy())
        return mean, covariance

    def score(
        self, parameters: Sequence[float], fit_depletion: bool = True
    ) -> dict[str, Any]:
        mean, simulation_covariance = self.predictive_distribution(
            parameters, fit_depletion
        )
        residual = mean - self.dataset.values
        total_covariance = self.dataset.covariance + simulation_covariance
        cholesky = np.linalg.cholesky(total_covariance)
        whitened = np.linalg.solve(cholesky, residual)
        chi2_total = float(whitened @ whitened)
        sign, logdet = np.linalg.slogdet(total_covariance)
        if sign <= 0.0:
            raise np.linalg.LinAlgError("predictive covariance is not positive definite")
        log_density = -0.5 * (
            chi2_total + logdet + self.dataset.size * np.log(2.0 * np.pi)
        )
        observational_chi2 = self.dataset.chi2(mean)
        return {
            "objective": float(-2.0 * log_density),
            "chi2_total": chi2_total,
            "chi2_observational": observational_chi2,
            "joint_log_predictive_density": float(log_density),
            "prediction_mean": mean,
            "simulation_covariance": simulation_covariance,
            "total_covariance": total_covariance,
            "raw_residual": residual,
            "whitened_residual": whitened,
        }

    def objective(
        self, parameters: Sequence[float], fit_depletion: bool = True
    ) -> float:
        try:
            value = self.score(parameters, fit_depletion)["objective"]
            return float(value) if np.isfinite(value) else 1.0e100
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ):
            return 1.0e100


def fit_dscd_trajectory_model(
    likelihood: DSCDTrajectoryLikelihood,
    *,
    fit_depletion: bool,
    optimizer_seeds: Sequence[int] = (17, 43),
    maxiter: int = 24,
) -> DSCDFitResult:
    bounds = PARAMETER_BOUNDS if fit_depletion else PARAMETER_BOUNDS[:2]
    objective = lambda p: likelihood.objective(p, fit_depletion)
    runs: list[tuple[np.ndarray, float, dict[str, Any]]] = []
    for seed in optimizer_seeds:
        result = differential_evolution(
            objective,
            bounds,
            seed=int(seed),
            maxiter=int(maxiter),
            popsize=6,
            tol=1.0e-5,
            atol=1.0e-6,
            polish=False,
            workers=1,
            updating="immediate",
        )
        runs.append(
            (
                np.asarray(result.x, dtype=float),
                float(result.fun),
                {
                    "seed": int(seed),
                    "objective": float(result.fun),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "evaluations": int(result.nfev),
                },
            )
        )
    start, start_value, _ = min(runs, key=lambda item: item[1])
    local = minimize(
        objective,
        start,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 800, "ftol": 1.0e-7, "xtol": 1.0e-6},
    )
    local_parameters = np.asarray(local.x, dtype=float)
    local_value = float(objective(local_parameters))
    if local_value <= start_value:
        parameters = local_parameters
        value = local_value
    else:
        parameters = start
        value = start_value
    score = likelihood.score(parameters, fit_depletion)
    names = PARAMETER_NAMES if fit_depletion else PARAMETER_NAMES[:2]
    active_bounds = tuple(
        name
        for name, parameter, (lower, upper) in zip(names, parameters, bounds)
        if min(abs(parameter - lower), abs(upper - parameter))
        <= max(1.0e-5 * (upper - lower), 1.0e-8)
    )
    return DSCDFitResult(
        parameters=parameters,
        objective=value,
        chi2_observational=float(score["chi2_observational"]),
        joint_log_predictive_density=float(score["joint_log_predictive_density"]),
        prediction_mean=np.asarray(score["prediction_mean"]),
        prediction_covariance=np.asarray(score["simulation_covariance"]),
        active_bounds=active_bounds,
        global_runs=tuple(item[2] for item in runs),
        local_message=str(local.message),
        fit_depletion=fit_depletion,
        optimization_seeds=tuple(int(seed) for seed in optimizer_seeds),
        trajectory_seeds=likelihood.trajectory_seeds,
    )


def whitened_jacobian_svd(
    likelihood: DSCDTrajectoryLikelihood,
    parameters: Sequence[float],
    *,
    fit_depletion: bool,
    relative_step: float = 2.0e-3,
) -> dict[str, Any]:
    """Finite-difference prediction Jacobian whitened by observational covariance."""
    p = np.asarray(parameters, dtype=float)
    bounds = PARAMETER_BOUNDS if fit_depletion else PARAMETER_BOUNDS[:2]
    columns: list[np.ndarray] = []
    steps: list[float] = []
    for index, (lower, upper) in enumerate(bounds):
        step = max(relative_step * (upper - lower), 1.0e-7)
        plus = p.copy()
        minus = p.copy()
        plus[index] = min(upper, plus[index] + step)
        minus[index] = max(lower, minus[index] - step)
        prediction_plus, _ = likelihood.predictive_distribution(plus, fit_depletion)
        prediction_minus, _ = likelihood.predictive_distribution(minus, fit_depletion)
        denominator = plus[index] - minus[index]
        columns.append((prediction_plus - prediction_minus) / denominator)
        steps.append(float(denominator))
    jacobian = np.column_stack(columns)
    whitened = np.linalg.solve(likelihood.dataset.cholesky(), jacobian)
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    positive = singular_values[singular_values > max(singular_values[0], 1.0) * 1e-10]
    condition = (
        float(positive[0] / positive[-1])
        if positive.size == len(bounds)
        else float("inf")
    )
    return {
        "parameter_names": list(
            PARAMETER_NAMES if fit_depletion else PARAMETER_NAMES[:2]
        ),
        "finite_difference_denominators": steps,
        "jacobian": jacobian.tolist(),
        "whitened_jacobian": whitened.tolist(),
        "singular_values": singular_values.tolist(),
        "rank": int(positive.size),
        "condition_number": condition,
    }


def fit_fixed_dscd_background(
    base_config: DSCDCosmologyConfig,
    dataset: BAODataset,
    trajectory_seeds: Sequence[int],
    *,
    quadrature_order: int = 48,
    omega_bounds: tuple[float, float] = (0.10, 0.60),
    profile_points: int = 25,
) -> dict[str, Any]:
    """Fit only theta and omega_m with every DSCD dynamic fixed.

    For a fixed omega_m, all BAO distances scale exactly as 1/theta.  The
    generalized least-squares optimum in that scale is analytic, leaving one
    bounded scalar search over omega_m.
    """
    inverse_covariance = np.linalg.inv(dataset.covariance)
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(omega_m: float) -> dict[str, Any]:
        key = float(np.round(omega_m, 12))
        if key in cache:
            return cache[key]
        config = replace(base_config, omega_m=float(omega_m)).validate()
        ensemble = run_ensemble(
            config,
            trajectory_seeds,
            variance_semantics="monte_carlo",
        )
        unit_samples = np.stack(
            [
                trajectory_prediction(
                    trajectory,
                    dataset,
                    theta=1.0,
                    quadrature_order=quadrature_order,
                )
                for trajectory in ensemble.trajectories
            ]
        )
        unit_mean = np.mean(unit_samples, axis=0)
        numerator = float(unit_mean @ inverse_covariance @ dataset.values)
        denominator = float(unit_mean @ inverse_covariance @ unit_mean)
        inverse_theta = numerator / denominator
        theta = float(
            np.clip(
                1.0 / inverse_theta,
                PARAMETER_BOUNDS[0][0],
                PARAMETER_BOUNDS[0][1],
            )
        )
        prediction_samples = unit_samples / theta
        prediction_mean = np.mean(prediction_samples, axis=0)
        chi2 = dataset.chi2(prediction_mean)
        item = {
            "omega_m": float(omega_m),
            "theta": theta,
            "chi2": float(chi2),
            "prediction_mean": prediction_mean,
            "prediction_samples": prediction_samples,
        }
        cache[key] = item
        return item

    bounded = minimize_scalar(
        lambda value: evaluate(float(value))["chi2"],
        bounds=omega_bounds,
        method="bounded",
        options={"xatol": 2.0e-4, "maxiter": 48},
    )
    grid = np.linspace(omega_bounds[0], omega_bounds[1], profile_points)
    profile = [evaluate(float(value)) for value in grid]
    candidates = profile + [evaluate(float(bounded.x))]
    best = min(candidates, key=lambda item: item["chi2"])

    sample_covariance = (
        np.atleast_2d(np.cov(best["prediction_samples"], rowvar=False, ddof=1))
        if len(trajectory_seeds) > 1
        else np.zeros_like(dataset.covariance)
    )
    total_covariance = dataset.covariance + sample_covariance
    residual = best["prediction_mean"] - dataset.values
    cholesky = np.linalg.cholesky(total_covariance)
    whitened = np.linalg.solve(cholesky, residual)
    sign, logdet = np.linalg.slogdet(total_covariance)
    if sign <= 0.0:
        raise np.linalg.LinAlgError("total predictive covariance is not positive definite")
    log_score = -0.5 * (
        float(whitened @ whitened)
        + logdet
        + dataset.size * np.log(2.0 * np.pi)
    )
    grid_best = min(profile, key=lambda item: item["chi2"])
    return {
        "parameter_names": ["theta", "omega_m"],
        "parameters": {
            "theta": float(best["theta"]),
            "omega_m": float(best["omega_m"]),
        },
        "parameter_vector": [float(best["theta"]), float(best["omega_m"])],
        "dscd_dynamics_frozen": True,
        "depletion_scale": 1.0,
        "chi2_observational": float(best["chi2"]),
        "joint_log_predictive_density": float(log_score),
        "prediction_mean": best["prediction_mean"].tolist(),
        "simulation_covariance": sample_covariance.tolist(),
        "trajectory_seeds": [int(seed) for seed in trajectory_seeds],
        "bounded_optimizer": {
            "omega_m": float(bounded.x),
            "chi2": float(bounded.fun),
            "success": bool(bounded.success),
            "message": str(bounded.message),
            "evaluations": int(bounded.nfev),
        },
        "grid_optimizer": {
            "omega_m": float(grid_best["omega_m"]),
            "theta": float(grid_best["theta"]),
            "chi2": float(grid_best["chi2"]),
            "points": int(profile_points),
        },
        "optimizer_prediction_agreement": float(
            np.linalg.norm(
                evaluate(float(bounded.x))["prediction_mean"]
                - grid_best["prediction_mean"]
            )
        ),
        "profile": {
            "omega_m": [float(item["omega_m"]) for item in profile],
            "theta": [float(item["theta"]) for item in profile],
            "chi2": [float(item["chi2"]) for item in profile],
            "delta_chi2": [
                float(item["chi2"] - best["chi2"]) for item in profile
            ],
        },
    }
