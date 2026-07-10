"""Pinned DESI DR1/DR2 BAO likelihood and flat-FRW distance utilities.

The embedded arrays are copied verbatim from CobayaSampler/bao_data commit
bb0c1c9009dc76d1391300e169e8df38fd1096db.  They are dimensionless distances
in units of the drag sound horizon.  BAO alone constrains
theta = H0 * r_d / c, not H0 and r_d separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Sequence

import numpy as np


PINNED_COMMIT = "bb0c1c9009dc76d1391300e169e8df38fd1096db"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/"
    f"{PINNED_COMMIT}/"
)

DR1_SOURCE = {
    "release": "DESI DR1 (Year 1)",
    "references": ["arXiv:2404.03000", "arXiv:2404.03001", "arXiv:2404.03002"],
    "mean_file": "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
    "covariance_file": "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",
    "mean_sha256": "dd2873a0b88459a491af3c0c0307ba059f62df9211d5b976760f310565a1be68",
    "covariance_sha256": "bbafa9074b51cf1a45e0d10e4f37db8c0e80a5d1d1788857abb7fc49fb21abcc",
}
DR2_SOURCE = {
    "release": "DESI DR2",
    "references": ["arXiv:2503.14738", "arXiv:2503.14739"],
    "mean_file": "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
    "covariance_file": "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
    "mean_sha256": "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585",
    "covariance_sha256": "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509",
}


@dataclass(frozen=True)
class Observable:
    redshift: float
    kind: str
    tracer: str

    @property
    def label(self) -> str:
        return f"{self.tracer}:{self.kind}@z={self.redshift:.3f}"


@dataclass(frozen=True)
class BAODataset:
    name: str
    observables: tuple[Observable, ...]
    values: np.ndarray
    covariance: np.ndarray
    source: dict[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        n = len(self.observables)
        if values.shape != (n,) or covariance.shape != (n, n):
            raise ValueError(f"{self.name}: inconsistent vector/covariance dimensions")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(covariance)):
            raise ValueError(f"{self.name}: non-finite official data")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-13):
            raise ValueError(f"{self.name}: covariance is not symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{self.name}: covariance is not positive definite") from exc
        condition = float(np.linalg.cond(covariance))
        if not np.isfinite(condition) or condition > 1.0e12:
            raise ValueError(f"{self.name}: ill-conditioned covariance ({condition:.3e})")
        values.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "covariance", covariance)

    @property
    def size(self) -> int:
        return len(self.observables)

    @property
    def redshifts(self) -> np.ndarray:
        return np.asarray([item.redshift for item in self.observables], dtype=float)

    @property
    def labels(self) -> list[str]:
        return [item.label for item in self.observables]

    def subset(self, indices: Sequence[int], name: str | None = None) -> "BAODataset":
        idx = np.asarray(indices, dtype=int)
        return BAODataset(
            name=name or f"{self.name} subset",
            observables=tuple(self.observables[i] for i in idx),
            values=self.values[idx],
            covariance=self.covariance[np.ix_(idx, idx)],
            source={**self.source, "parent": self.name, "indices": idx.tolist()},
        )

    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.covariance)

    def residual(self, prediction: Sequence[float]) -> np.ndarray:
        prediction_array = np.asarray(prediction, dtype=float)
        if prediction_array.shape != self.values.shape:
            raise ValueError(f"{self.name}: prediction has shape {prediction_array.shape}")
        if not np.all(np.isfinite(prediction_array)):
            raise ValueError(f"{self.name}: prediction contains non-finite values")
        return prediction_array - self.values

    def whiten(self, residual: Sequence[float]) -> np.ndarray:
        return np.linalg.solve(self.cholesky(), np.asarray(residual, dtype=float))

    def chi2(self, prediction: Sequence[float]) -> float:
        whitened = self.whiten(self.residual(prediction))
        return float(whitened @ whitened)

    def diagnostics(self, prediction: Sequence[float]) -> dict[str, object]:
        prediction_array = np.asarray(prediction, dtype=float)
        residual = self.residual(prediction_array)
        marginal = residual / np.sqrt(np.diag(self.covariance))
        whitened = self.whiten(residual)
        return {
            "chi2": float(whitened @ whitened),
            "predictions": prediction_array.tolist(),
            "raw_residuals": residual.tolist(),
            "marginal_standardized_residuals": marginal.tolist(),
            "cholesky_whitened_residuals": whitened.tolist(),
            "labels": self.labels,
        }


def _block_covariance(size: int, blocks: Iterable[tuple[Sequence[int], np.ndarray]]) -> np.ndarray:
    covariance = np.zeros((size, size), dtype=float)
    for indices, block in blocks:
        idx = np.asarray(indices, dtype=int)
        covariance[np.ix_(idx, idx)] = np.asarray(block, dtype=float)
    return covariance


DR1_OBSERVABLES = (
    Observable(0.295, "DV", "BGS"),
    Observable(0.510, "DM", "LRG1"),
    Observable(0.510, "DH", "LRG1"),
    Observable(0.706, "DM", "LRG2"),
    Observable(0.706, "DH", "LRG2"),
    Observable(0.930, "DM", "LRG3+ELG1"),
    Observable(0.930, "DH", "LRG3+ELG1"),
    Observable(1.317, "DM", "ELG2"),
    Observable(1.317, "DH", "ELG2"),
    Observable(1.491, "DV", "QSO"),
    Observable(2.330, "DM", "Lya"),
    Observable(2.330, "DH", "Lya"),
)
DR1_VALUES = np.asarray(
    [
        7.92512927,
        13.62003080,
        20.98334647,
        16.84645313,
        20.07872919,
        21.70841761,
        17.87612922,
        27.78720817,
        13.82372285,
        26.07217182,
        39.70838281,
        8.52256583,
    ]
)
DR1_COVARIANCE = _block_covariance(
    12,
    [
        ([0], [[2.27230845e-02]]),
        ([1, 2], [[6.34662240e-02, -6.85337250e-02], [-6.85337250e-02, 3.72968756e-01]]),
        ([3, 4], [[1.01975713e-01, -7.99403059e-02], [-7.99403059e-02, 3.54449156e-01]]),
        ([5, 6], [[7.95675235e-02, -3.80110101e-02], [-3.80110101e-02, 1.19935683e-01]]),
        ([7, 8], [[4.76569857e-01, -1.29405759e-01], [-1.29405759e-01, 1.78270498e-01]]),
        ([9], [[4.47134991e-01]]),
        ([10, 11], [[8.89752928e-01, -7.69477120e-02], [-7.69477120e-02, 2.91860447e-02]]),
    ],
)

DR2_OBSERVABLES = (
    Observable(0.295, "DV", "BGS"),
    Observable(0.510, "DM", "LRG1"),
    Observable(0.510, "DH", "LRG1"),
    Observable(0.706, "DM", "LRG2"),
    Observable(0.706, "DH", "LRG2"),
    Observable(0.934, "DM", "LRG3+ELG1"),
    Observable(0.934, "DH", "LRG3+ELG1"),
    Observable(1.321, "DM", "ELG2"),
    Observable(1.321, "DH", "ELG2"),
    Observable(1.484, "DM", "QSO"),
    Observable(1.484, "DH", "QSO"),
    Observable(2.330, "DH", "Lya"),
    Observable(2.330, "DM", "Lya"),
)
DR2_VALUES = np.asarray(
    [
        7.94167639,
        13.58758434,
        21.86294686,
        17.35069094,
        19.45534918,
        21.57563956,
        17.64149464,
        27.60085612,
        14.17602155,
        30.51190063,
        12.81699964,
        8.631545674846294,
        38.988973961958784,
    ]
)
DR2_COVARIANCE = _block_covariance(
    13,
    [
        ([0], [[5.78998687e-03]]),
        ([1, 2], [[2.83473742e-02, -3.26062007e-02], [-3.26062007e-02, 1.83928040e-01]]),
        ([3, 4], [[3.23752442e-02, -2.37445646e-02], [-2.37445646e-02, 1.11469198e-01]]),
        ([5, 6], [[2.61732816e-02, -1.12938006e-02], [-1.12938006e-02, 4.04183878e-02]]),
        ([7, 8], [[1.05336516e-01, -2.90308418e-02], [-2.90308418e-02, 5.04233092e-02]]),
        ([9, 10], [[5.83020277e-01, -1.95215562e-01], [-1.95215562e-01, 2.68336193e-01]]),
        ([11, 12], [[1.02136194e-02, -2.31395216e-02], [-2.31395216e-02, 2.82685779e-01]]),
    ],
)

DR1 = BAODataset(
    "DESI_DR1",
    DR1_OBSERVABLES,
    DR1_VALUES,
    DR1_COVARIANCE,
    {**DR1_SOURCE, "mean_url": SOURCE_ROOT + DR1_SOURCE["mean_file"], "covariance_url": SOURCE_ROOT + DR1_SOURCE["covariance_file"]},
)
DR2 = BAODataset(
    "DESI_DR2",
    DR2_OBSERVABLES,
    DR2_VALUES,
    DR2_COVARIANCE,
    {**DR2_SOURCE, "mean_url": SOURCE_ROOT + DR2_SOURCE["mean_file"], "covariance_url": SOURCE_ROOT + DR2_SOURCE["covariance_file"]},
)


@lru_cache(maxsize=None)
def gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache immutable quadrature rules reused by every optimizer evaluation."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def bao_distances(
    redshifts: Sequence[float],
    theta: float,
    omega_m: float,
    f_de: Callable[[np.ndarray], np.ndarray],
    omega_r: float = 0.0,
    quadrature_order: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (DM/rd, DH/rd, DV/rd) for a flat FRW background."""
    z = np.asarray(redshifts, dtype=float)
    if z.ndim != 1 or np.any(~np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("redshifts must be a finite non-negative vector")
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be finite and positive")
    if not np.isfinite(omega_m) or not 0.0 < omega_m < 1.0:
        raise ValueError("omega_m must lie strictly between zero and one")
    if not np.isfinite(omega_r) or omega_r < 0.0:
        raise ValueError("omega_r must be finite and non-negative")
    omega_de = 1.0 - omega_m - omega_r
    if omega_de < 0.0:
        raise ValueError("flat-background Omega_DE must be non-negative")
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")

    nodes, weights = gauss_legendre(quadrature_order)

    def expansion_squared(x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        dark_energy = np.asarray(f_de(x_array), dtype=float)
        if dark_energy.shape != x_array.shape:
            dark_energy = np.broadcast_to(dark_energy, x_array.shape)
        e2 = (
            omega_m * (1.0 + x_array) ** 3
            + omega_r * (1.0 + x_array) ** 4
            + omega_de * dark_energy
        )
        if np.any(~np.isfinite(e2)) or np.any(e2 <= 0.0):
            raise ValueError("model produced non-finite or non-positive E(z)^2")
        return e2

    dh = 1.0 / (theta * np.sqrt(expansion_squared(z)))
    if z.size:
        integration_nodes = 0.5 * z[:, None] * (nodes[None, :] + 1.0)
        integration_weights = 0.5 * z[:, None] * weights[None, :]
        dm = np.sum(integration_weights / np.sqrt(expansion_squared(integration_nodes)), axis=1) / theta
    else:
        dm = np.empty(0, dtype=float)
    dv = np.cbrt(z * dm * dm * dh)
    if np.any(~np.isfinite(dm)) or np.any(~np.isfinite(dh)) or np.any(~np.isfinite(dv)):
        raise ValueError("distance calculation produced non-finite values")
    return dm, dh, dv


def bao_distances_from_expansion(
    redshifts: Sequence[float],
    theta: float,
    expansion: Callable[[np.ndarray], np.ndarray],
    quadrature_order: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return BAO distances from a trajectory-derived E(z)=H(z)/H0.

    Unlike ``bao_distances``, this interface does not reconstruct E(z) from a
    dark-energy equation-of-state ansatz.  It consumes the expansion history
    generated by a complete dynamical trajectory.
    """
    z = np.asarray(redshifts, dtype=float)
    if z.ndim != 1 or np.any(~np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("redshifts must be a finite non-negative vector")
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be finite and positive")
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")

    def checked_expansion(x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        result = np.asarray(expansion(x_array), dtype=float)
        if result.shape != x_array.shape:
            result = np.broadcast_to(result, x_array.shape)
        if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("trajectory produced non-finite or non-positive E(z)")
        return result

    nodes, weights = gauss_legendre(quadrature_order)
    e_at_z = checked_expansion(z)
    dh = 1.0 / (theta * e_at_z)
    if z.size:
        integration_nodes = 0.5 * z[:, None] * (nodes[None, :] + 1.0)
        integration_weights = 0.5 * z[:, None] * weights[None, :]
        dm = (
            np.sum(
                integration_weights / checked_expansion(integration_nodes),
                axis=1,
            )
            / theta
        )
    else:
        dm = np.empty(0, dtype=float)
    dv = np.cbrt(z * dm * dm * dh)
    if np.any(~np.isfinite(dm)) or np.any(~np.isfinite(dh)) or np.any(~np.isfinite(dv)):
        raise ValueError("trajectory distance calculation produced non-finite values")
    return dm, dh, dv


def predict_dataset(
    dataset: BAODataset,
    theta: float,
    omega_m: float,
    f_de: Callable[[np.ndarray], np.ndarray],
    omega_r: float = 0.0,
    quadrature_order: int = 64,
) -> np.ndarray:
    unique_z, inverse = np.unique(dataset.redshifts, return_inverse=True)
    dm, dh, dv = bao_distances(unique_z, theta, omega_m, f_de, omega_r, quadrature_order)
    lookup = {"DM": dm, "DH": dh, "DV": dv}
    return np.asarray(
        [lookup[item.kind][inverse[i]] for i, item in enumerate(dataset.observables)],
        dtype=float,
    )


def predict_dataset_from_expansion(
    dataset: BAODataset,
    theta: float,
    expansion: Callable[[np.ndarray], np.ndarray],
    quadrature_order: int = 64,
) -> np.ndarray:
    """Predict a BAO vector from an emergent trajectory expansion history."""
    unique_z, inverse = np.unique(dataset.redshifts, return_inverse=True)
    dm, dh, dv = bao_distances_from_expansion(
        unique_z,
        theta,
        expansion,
        quadrature_order=quadrature_order,
    )
    lookup = {"DM": dm, "DH": dh, "DV": dv}
    return np.asarray(
        [lookup[item.kind][inverse[i]] for i, item in enumerate(dataset.observables)],
        dtype=float,
    )


def conditional_target_diagnostics(
    dataset: BAODataset,
    prediction: Sequence[float],
    train_indices: Sequence[int],
    target_indices: Sequence[int],
) -> dict[str, object]:
    """Evaluate target residuals conditional on same-release training residuals."""
    train = np.asarray(train_indices, dtype=int)
    target = np.asarray(target_indices, dtype=int)
    if set(train.tolist()) & set(target.tolist()):
        raise ValueError("training and target indices must be disjoint")
    residual = dataset.residual(prediction)
    c_ll = dataset.covariance[np.ix_(train, train)]
    c_tt = dataset.covariance[np.ix_(target, target)]
    c_tl = dataset.covariance[np.ix_(target, train)]
    correction = c_tl @ np.linalg.solve(c_ll, residual[train])
    conditional_residual = residual[target] - correction
    conditional_covariance = c_tt - c_tl @ np.linalg.solve(c_ll, c_tl.T)
    cholesky = np.linalg.cholesky(conditional_covariance)
    whitened = np.linalg.solve(cholesky, conditional_residual)
    marginal = conditional_residual / np.sqrt(np.diag(conditional_covariance))
    return {
        "chi2": float(whitened @ whitened),
        "conditional_residuals": conditional_residual.tolist(),
        "conditional_covariance": conditional_covariance.tolist(),
        "marginal_standardized_residuals": marginal.tolist(),
        "cholesky_whitened_residuals": whitened.tolist(),
        "cross_covariance_max_abs": float(np.max(np.abs(c_tl))) if c_tl.size else 0.0,
        "labels": [dataset.labels[i] for i in target],
    }
