"""Cosmological model registry and bounded continuous fitting for clean DESI BAO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from desi_bao_likelihood import BAODataset, gauss_legendre, predict_dataset


LARGE_OBJECTIVE = 1.0e100


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    description: str
    f_de_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    w_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    constraint: Callable[[np.ndarray], bool] = lambda _parameters: True
    regularity_note: str = ""

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def validate_parameters(self, parameters: Sequence[float]) -> np.ndarray:
        p = np.asarray(parameters, dtype=float)
        if p.shape != (self.parameter_count,) or np.any(~np.isfinite(p)):
            raise ValueError(f"{self.name}: invalid parameter vector")
        for value, (lower, upper), parameter_name in zip(p, self.bounds, self.parameter_names):
            if value < lower or value > upper:
                raise ValueError(f"{self.name}: {parameter_name} outside declared bounds")
        if not self.constraint(p):
            raise ValueError(f"{self.name}: parameter constraint violated")
        return p

    def f_de(self, redshift: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
        p = self.validate_parameters(parameters)
        result = np.asarray(self.f_de_function(np.asarray(redshift, dtype=float), p), dtype=float)
        if result.shape != np.asarray(redshift).shape:
            result = np.broadcast_to(result, np.asarray(redshift).shape)
        if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(f"{self.name}: invalid dark-energy density factor")
        return result

    def w(self, redshift: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
        p = self.validate_parameters(parameters)
        return np.asarray(self.w_function(np.asarray(redshift, dtype=float), p), dtype=float)

    def predict(
        self,
        dataset: BAODataset,
        parameters: Sequence[float],
        omega_r: float = 0.0,
        quadrature_order: int = 64,
    ) -> np.ndarray:
        p = self.validate_parameters(parameters)
        theta, omega_m = p[:2]
        return predict_dataset(
            dataset,
            theta,
            omega_m,
            lambda z: self.f_de(z, p),
            omega_r=omega_r,
            quadrature_order=quadrature_order,
        )


def _ones(z: np.ndarray, _p: np.ndarray) -> np.ndarray:
    return np.ones_like(z, dtype=float)


def _minus_one(z: np.ndarray, _p: np.ndarray) -> np.ndarray:
    return np.full_like(z, -1.0, dtype=float)


def _constant_w_fde(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.exp(3.0 * (1.0 + p[2]) * np.log1p(z))


def _constant_w(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.full_like(z, p[2], dtype=float)


def _cpl_fde(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    w0, wa = p[2], p[3]
    log_f = 3.0 * (1.0 + w0 + wa) * np.log1p(z) - 3.0 * wa * z / (1.0 + z)
    if np.any(np.abs(log_f) > 700.0):
        raise ValueError("CPL density overflow")
    return np.exp(log_f)


def _cpl_w(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[2] + p[3] * z / (1.0 + z)


def _cpl_nonphantom_constraint(p: np.ndarray) -> bool:
    # For z >= 0, u=z/(1+z) lies in [0,1), so the minimum is at an endpoint.
    return bool(p[2] >= -1.0 and p[2] + p[3] >= -1.0)


def _monotone_one_fde(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Generic one-shape monotone density control, not a DSCD model."""
    return np.exp(p[2] * z / (1.0 + z))


def _monotone_one_w(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -1.0 + p[2] / (3.0 * (1.0 + z))


def _cas_log_fde(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    amplitude = 1.0 + p[2]
    denominator = 1.0 + 1.5 * amplitude * np.log1p(z)
    if np.any(denominator <= 0.0):
        raise ValueError("PHEN_LOG denominator is non-positive")
    return denominator * denominator


def _cas_log_w(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    amplitude = 1.0 + p[2]
    return -1.0 + amplitude / (1.0 + 1.5 * amplitude * np.log1p(z))


def _cas_poly_integral(z: np.ndarray, amplitude: float, gamma: float) -> np.ndarray:
    """Integral of (1+w)/(1+z), with a stable special case near gamma=4."""
    t = 1.0 + z
    a = 1.0 - gamma / 4.0
    denominator = 1.0 + gamma * z * (z + 2.0) / 4.0
    if np.any(denominator <= 0.0):
        raise ValueError("PHEN_POLY denominator is non-positive")
    if abs(a) > 1.0e-5:
        return amplitude / a * (np.log(t) - 0.5 * np.log(denominator))

    # At gamma=4 the integrand is amplitude/t^3.  Gauss-Legendre integration
    # avoids cancellation in a narrow neighborhood of this removable limit.
    nodes, weights = gauss_legendre(24)
    x = 0.5 * z[..., None] * (nodes + 1.0)
    scaled_weights = 0.5 * z[..., None] * weights
    integrand = amplitude / (
        (1.0 + gamma * x * (x + 2.0) / 4.0) * (1.0 + x)
    )
    return np.sum(scaled_weights * integrand, axis=-1)


def _cas_poly_fde(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    integral = _cas_poly_integral(z, 1.0 + p[2], p[3])
    log_f = 3.0 * integral
    if np.any(np.abs(log_f) > 700.0):
        raise ValueError("PHEN_POLY density overflow")
    return np.exp(log_f)


def _cas_poly_w(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -1.0 + (1.0 + p[2]) / (1.0 + p[3] * z * (z + 2.0) / 4.0)


BASE_BOUNDS = ((0.020, 0.045), (0.10, 0.60))
MODELS: dict[str, ModelSpec] = {
    "LCDM": ModelSpec(
        "LCDM",
        "APS-reproduction",
        ("theta", "omega_m"),
        BASE_BOUNDS,
        "Flat LambdaCDM; w=-1.",
        _ones,
        _minus_one,
    ),
    "WCDM": ModelSpec(
        "WCDM",
        "APS-reproduction",
        ("theta", "omega_m", "w0"),
        BASE_BOUNDS + ((-3.0, 1.0),),
        "Unconstrained constant-w control.",
        _constant_w_fde,
        _constant_w,
    ),
    "WCDM_NONPHANTOM": ModelSpec(
        "WCDM_NONPHANTOM",
        "APS-reproduction",
        ("theta", "omega_m", "w0"),
        BASE_BOUNDS + ((-1.0, 1.0),),
        "Constant-w model with the non-phantom inequality w0>=-1.",
        _constant_w_fde,
        _constant_w,
        regularity_note="Standard information criteria are suppressed at the w0=-1 boundary.",
    ),
    "CPL": ModelSpec(
        "CPL",
        "APS-reproduction",
        ("theta", "omega_m", "w0", "wa"),
        BASE_BOUNDS + ((-3.0, 1.0), (-8.0, 8.0)),
        "Unconstrained CPL w(z)=w0+wa*z/(1+z).",
        _cpl_fde,
        _cpl_w,
    ),
    "CPL_NONPHANTOM": ModelSpec(
        "CPL_NONPHANTOM",
        "prior-isolation",
        ("theta", "omega_m", "w0", "wa"),
        BASE_BOUNDS + ((-1.0, 1.0), (-4.0, 8.0)),
        "CPL constrained to w(z)>=-1 for every z>=0.",
        _cpl_fde,
        _cpl_w,
        constraint=_cpl_nonphantom_constraint,
        regularity_note="Inequality-constrained model; ordinary AIC/AICc may be nonregular.",
    ),
    "MONO_ONE": ModelSpec(
        "MONO_ONE",
        "prior-isolation",
        ("theta", "omega_m", "monotone_amplitude"),
        BASE_BOUNDS + ((0.0, 2.0),),
        "Generic one-shape monotone non-phantom background control; not DSCD.",
        _monotone_one_fde,
        _monotone_one_w,
        regularity_note="Nests LambdaCDM at a non-negative amplitude boundary.",
    ),
    "PHEN_LOG": ModelSpec(
        "PHEN_LOG",
        "phenomenological-control",
        ("theta", "omega_m", "w0"),
        BASE_BOUNDS + ((-1.0, 1.0),),
        "Historical logarithmic mean-field ansatz; not a DSCD trajectory.",
        _cas_log_fde,
        _cas_log_w,
        regularity_note="At w0=-1 this nests LambdaCDM on a boundary.",
    ),
    "PHEN_POLY": ModelSpec(
        "PHEN_POLY",
        "phenomenological-control",
        ("theta", "omega_m", "w0", "gamma"),
        BASE_BOUNDS + ((-1.0, 1.0), (0.0, 30.0)),
        "Historical polynomial mean-field ansatz; not a DSCD trajectory.",
        _cas_poly_fde,
        _cas_poly_w,
        regularity_note="Gamma is unidentifiable at w0=-1; regular criteria are then invalid.",
    ),
}

DEFAULT_MODEL_ORDER = tuple(MODELS)


@dataclass
class FitResult:
    model_name: str
    parameters: np.ndarray
    chi2: float
    converged: bool
    global_runs: list[dict[str, object]]
    local_message: str
    active_bounds: list[str]
    hessian_rank: int
    hessian_eigenvalues: list[float]
    regular_interior_fit: bool
    warnings: list[str]

    def to_dict(self, model: ModelSpec) -> dict[str, object]:
        return {
            "model": self.model_name,
            "family": model.family,
            "description": model.description,
            "parameter_names": list(model.parameter_names),
            "parameters": {
                name: float(value) for name, value in zip(model.parameter_names, self.parameters)
            },
            "parameter_vector": self.parameters.tolist(),
            "parameter_bounds": {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in zip(model.parameter_names, model.bounds)
            },
            "chi2": float(self.chi2),
            "converged": bool(self.converged),
            "global_runs": self.global_runs,
            "local_message": self.local_message,
            "active_bounds": self.active_bounds,
            "hessian_rank": int(self.hessian_rank),
            "hessian_eigenvalues": self.hessian_eigenvalues,
            "regular_interior_fit": bool(self.regular_interior_fit),
            "regularity_note": model.regularity_note,
            "warnings": self.warnings,
        }


def objective_function(
    parameters: Sequence[float],
    model: ModelSpec,
    dataset: BAODataset,
    omega_r: float = 0.0,
    quadrature_order: int = 64,
) -> float:
    try:
        prediction = model.predict(dataset, parameters, omega_r, quadrature_order)
        value = dataset.chi2(prediction)
        return value if np.isfinite(value) else LARGE_OBJECTIVE
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return LARGE_OBJECTIVE


def _numerical_hessian(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    bounds: Sequence[tuple[float, float]],
) -> np.ndarray:
    n = point.size
    hessian = np.zeros((n, n), dtype=float)
    steps = np.asarray([max((upper - lower) * 2.0e-4, 1.0e-6) for lower, upper in bounds])
    f0 = function(point)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        plus = np.clip(point + ei, [b[0] for b in bounds], [b[1] for b in bounds])
        minus = np.clip(point - ei, [b[0] for b in bounds], [b[1] for b in bounds])
        if plus[i] == minus[i]:
            continue
        hessian[i, i] = (function(plus) - 2.0 * f0 + function(minus)) / (
            0.25 * (plus[i] - minus[i]) ** 2
        )
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = steps[j]
            pp = np.clip(point + ei + ej, [b[0] for b in bounds], [b[1] for b in bounds])
            pm = np.clip(point + ei - ej, [b[0] for b in bounds], [b[1] for b in bounds])
            mp = np.clip(point - ei + ej, [b[0] for b in bounds], [b[1] for b in bounds])
            mm = np.clip(point - ei - ej, [b[0] for b in bounds], [b[1] for b in bounds])
            denominator = (pp[i] - mp[i]) * (pp[j] - pm[j])
            if denominator != 0.0:
                hessian[i, j] = hessian[j, i] = (
                    function(pp) - function(pm) - function(mp) + function(mm)
                ) / denominator
    return 0.5 * (hessian + hessian.T)


def fit_model(
    model: ModelSpec,
    dataset: BAODataset,
    *,
    mode: str = "quick",
    omega_r: float = 0.0,
    quadrature_order: int | None = None,
    seeds: Sequence[int] | None = None,
    maxiter: int | None = None,
) -> FitResult:
    if mode not in {"quick", "publication"}:
        raise ValueError("mode must be 'quick' or 'publication'")
    order = quadrature_order or (48 if mode == "quick" else 96)
    selected_seeds = tuple(seeds or ((17, 43) if mode == "quick" else (17, 43, 101)))
    iterations = int(maxiter or (70 if mode == "quick" else 180))
    objective = lambda p: objective_function(p, model, dataset, omega_r, order)

    runs: list[tuple[np.ndarray, float, dict[str, object]]] = []
    for seed in selected_seeds:
        result = differential_evolution(
            objective,
            model.bounds,
            seed=int(seed),
            maxiter=iterations,
            popsize=8 if mode == "quick" else 12,
            tol=2.0e-7 if mode == "quick" else 2.0e-9,
            atol=1.0e-8,
            polish=False,
            workers=1,
            updating="immediate",
        )
        metadata = {
            "seed": int(seed),
            "chi2": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
        }
        runs.append((np.asarray(result.x, dtype=float), float(result.fun), metadata))

    start, start_value, _ = min(runs, key=lambda item: item[1])
    local = minimize(
        objective,
        start,
        method="Powell",
        bounds=model.bounds,
        options={
            "xtol": 1.0e-9 if mode == "publication" else 1.0e-7,
            "ftol": 1.0e-10 if mode == "publication" else 1.0e-8,
            "maxiter": 4000,
        },
    )
    local_parameters = np.asarray(local.x, dtype=float)
    local_value = float(objective(local_parameters))
    if local_value <= start_value:
        parameters = local_parameters
        chi2 = local_value
    else:
        parameters = start.copy()
        chi2 = float(start_value)
    ranges = np.asarray([upper - lower for lower, upper in model.bounds])
    active_bounds = [
        name
        for name, value, (lower, upper), width in zip(
            model.parameter_names, parameters, model.bounds, ranges
        )
        if min(abs(value - lower), abs(upper - value)) <= max(1.0e-5 * width, 1.0e-8)
    ]

    hessian = _numerical_hessian(objective, parameters, model.bounds)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
    except np.linalg.LinAlgError:
        eigenvalues = np.full(model.parameter_count, np.nan)
    finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    scale = max(float(np.max(np.abs(finite_eigenvalues))) if finite_eigenvalues.size else 0.0, 1.0)
    rank = int(np.sum(finite_eigenvalues > scale * 1.0e-7))

    seed_values = np.asarray([item[1] for item in runs])
    warnings: list[str] = []
    if float(np.ptp(seed_values)) > 1.0e-4:
        warnings.append("Global optimizer seeds disagree by more than 1e-4 in chi2.")
    if active_bounds:
        warnings.append("Best fit reaches declared bound(s): " + ", ".join(active_bounds))
    if rank < model.parameter_count:
        warnings.append("Numerical Hessian is rank deficient or non-positive.")
    if model.name == "PHEN_POLY" and abs(parameters[2] + 1.0) < 1.0e-4:
        warnings.append("PHEN_POLY gamma is unidentifiable on the LambdaCDM boundary.")
        rank = min(rank, model.parameter_count - 1)

    regular = not active_bounds and rank == model.parameter_count and model.name not in {
        "CPL_NONPHANTOM"
    }
    converged = bool(
        chi2 < LARGE_OBJECTIVE
        and np.all(np.isfinite(parameters))
        and float(np.ptp(seed_values)) <= 1.0e-3
        and (local.success or "Optimization terminated successfully" in str(local.message))
    )
    return FitResult(
        model_name=model.name,
        parameters=parameters,
        chi2=chi2,
        converged=converged,
        global_runs=[item[2] for item in runs],
        local_message=str(local.message),
        active_bounds=active_bounds,
        hessian_rank=rank,
        hessian_eigenvalues=[float(value) for value in eigenvalues],
        regular_interior_fit=regular,
        warnings=warnings,
    )


def profile_parameter(
    model: ModelSpec,
    dataset: BAODataset,
    fit: FitResult,
    parameter_name: str,
    *,
    points: int = 11,
    omega_r: float = 0.0,
    quadrature_order: int = 64,
) -> dict[str, object]:
    index = model.parameter_names.index(parameter_name)
    lower, upper = model.bounds[index]
    grid = np.linspace(lower, upper, points)
    free_indices = [i for i in range(model.parameter_count) if i != index]
    profile: list[float | None] = []

    for fixed in grid:
        def reconstruct(free: np.ndarray) -> np.ndarray:
            full = fit.parameters.copy()
            full[index] = fixed
            full[free_indices] = free
            return full

        free_bounds = [model.bounds[i] for i in free_indices]
        if free_indices:
            local = minimize(
                lambda free: objective_function(
                    reconstruct(free), model, dataset, omega_r, quadrature_order
                ),
                fit.parameters[free_indices],
                method="Powell",
                bounds=free_bounds,
                options={"maxiter": 1500, "ftol": 1.0e-7, "xtol": 1.0e-6},
            )
            value = objective_function(
                reconstruct(np.asarray(local.x)), model, dataset, omega_r, quadrature_order
            )
        else:
            value = objective_function(
                np.asarray([fixed]), model, dataset, omega_r, quadrature_order
            )
        profile.append(float(value) if value < LARGE_OBJECTIVE else None)

    return {
        "parameter": parameter_name,
        "grid": grid.tolist(),
        "chi2": profile,
        "delta_chi2": [
            (value - fit.chi2) if value is not None else None for value in profile
        ],
    }
