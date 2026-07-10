"""State and trajectory containers for the coupled DSCD+FLRW system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


@dataclass
class DSCDCosmologyState:
    """Complete Markov state, including memory and phase."""

    step: int
    N: float
    rho_c: np.ndarray
    x: np.ndarray
    y: np.ndarray
    structure: np.ndarray
    structure_prev1: np.ndarray
    structure_prev2: np.ndarray
    phase: np.ndarray
    dtau_prev1: float
    dtau_prev2: float

    def validate(self) -> "DSCDCosmologyState":
        size = np.asarray(self.rho_c).size
        if size < 1:
            raise ValueError("state must contain at least one cell")
        for name in (
            "rho_c",
            "x",
            "y",
            "structure",
            "structure_prev1",
            "structure_prev2",
            "phase",
        ):
            setattr(self, name, _finite_vector(name, getattr(self, name), size))
        if np.any(self.rho_c < 0.0) or np.any(self.x < 0.0) or np.any(self.y < 0.0):
            raise ValueError("density and activity states must be non-negative")
        if not np.isfinite(self.N):
            raise ValueError("N must be finite")
        if self.dtau_prev1 < 0.0 or self.dtau_prev2 < 0.0:
            raise ValueError("proper-time history must be non-negative")
        return self

    @property
    def n_cells(self) -> int:
        return int(self.rho_c.size)

    def copy(self) -> "DSCDCosmologyState":
        return DSCDCosmologyState(
            step=int(self.step),
            N=float(self.N),
            rho_c=self.rho_c.copy(),
            x=self.x.copy(),
            y=self.y.copy(),
            structure=self.structure.copy(),
            structure_prev1=self.structure_prev1.copy(),
            structure_prev2=self.structure_prev2.copy(),
            phase=self.phase.copy(),
            dtau_prev1=float(self.dtau_prev1),
            dtau_prev2=float(self.dtau_prev2),
        )


@dataclass(frozen=True)
class EventRequests:
    """Requested events before species/resource funding caps."""

    xy: np.ndarray
    xx: np.ndarray
    explosion: np.ndarray
    bonds: np.ndarray

    @classmethod
    def zeros(cls, n_cells: int) -> "EventRequests":
        zero = np.zeros(n_cells, dtype=int)
        return cls(zero.copy(), zero.copy(), zero.copy(), zero.copy())


@dataclass(frozen=True)
class StepDiagnostics:
    dtau: float
    expansion: float
    requested_xy: np.ndarray
    requested_xx: np.ndarray
    funded_xy: np.ndarray
    funded_xx: np.ndarray
    funded_explosion: np.ndarray
    funded_bonds: np.ndarray
    leakage: np.ndarray
    debit: np.ndarray
    ripple: np.ndarray
    regimes: np.ndarray
    pressure_c: float
    w_interval: float
    continuity_residual: float
    transport_x_residual: float
    transport_y_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtau": float(self.dtau),
            "expansion": float(self.expansion),
            "requested_xy": self.requested_xy.tolist(),
            "requested_xx": self.requested_xx.tolist(),
            "funded_xy": self.funded_xy.tolist(),
            "funded_xx": self.funded_xx.tolist(),
            "funded_explosion": self.funded_explosion.tolist(),
            "funded_bonds": self.funded_bonds.tolist(),
            "leakage": self.leakage.tolist(),
            "debit": self.debit.tolist(),
            "ripple": self.ripple.tolist(),
            "regimes": self.regimes.tolist(),
            "pressure_c": float(self.pressure_c),
            "w_interval": float(self.w_interval),
            "continuity_residual": float(self.continuity_residual),
            "transport_x_residual": float(self.transport_x_residual),
            "transport_y_residual": float(self.transport_y_residual),
        }


@dataclass(frozen=True)
class DSCDCosmologyTrajectory:
    """One stochastic DSCD+GR path sampled on an e-fold lattice."""

    N: np.ndarray
    expansion: np.ndarray
    rho_c: np.ndarray
    pressure_c: np.ndarray
    depletion_rate: np.ndarray
    w_interval: np.ndarray
    ripple: np.ndarray
    x_total: np.ndarray
    y_total: np.ndarray
    structure_total: np.ndarray
    regime_fractions: np.ndarray
    continuity_residual: np.ndarray
    friedmann_residual: np.ndarray
    dark_coefficient: float
    closure_converged: bool
    seed: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        length = np.asarray(self.N).size
        point_arrays = (
            self.expansion,
            self.rho_c,
            self.pressure_c,
            self.ripple,
            self.x_total,
            self.y_total,
            self.structure_total,
            self.friedmann_residual,
        )
        if length < 2 or any(np.asarray(item).shape != (length,) for item in point_arrays):
            raise ValueError("trajectory point arrays have inconsistent lengths")
        interval_length = length - 1
        if np.asarray(self.depletion_rate).shape != (interval_length,):
            raise ValueError("depletion_rate must be interval-valued")
        if np.asarray(self.w_interval).shape != (interval_length,):
            raise ValueError("w_interval must be interval-valued")
        if np.asarray(self.continuity_residual).shape != (interval_length,):
            raise ValueError("continuity_residual must be interval-valued")
        if np.asarray(self.regime_fractions).shape != (interval_length, 3):
            raise ValueError("regime_fractions must have shape (steps,3)")
        if np.any(np.diff(self.N) <= 0.0):
            raise ValueError("trajectory N grid must increase strictly")
        if np.any(self.expansion <= 0.0) or np.any(self.rho_c <= 0.0):
            raise ValueError("trajectory expansion and density must remain positive")

    @property
    def redshift(self) -> np.ndarray:
        return np.exp(-self.N) - 1.0

    @property
    def f_de(self) -> np.ndarray:
        return self.rho_c / self.rho_c[-1]

    def expansion_at(self, redshift: np.ndarray | float) -> np.ndarray:
        z = np.asarray(redshift, dtype=float)
        if np.any(~np.isfinite(z)) or np.any(z < 0.0):
            raise ValueError("redshift must be finite and non-negative")
        source_z = self.redshift[::-1]
        if np.any(z < source_z[0] - 1e-12) or np.any(z > source_z[-1] + 1e-12):
            raise ValueError("redshift lies outside the simulated trajectory")
        return np.exp(np.interp(z, source_z, np.log(self.expansion[::-1])))

    def f_de_at(self, redshift: np.ndarray | float) -> np.ndarray:
        z = np.asarray(redshift, dtype=float)
        source_z = self.redshift[::-1]
        if np.any(z < source_z[0] - 1e-12) or np.any(z > source_z[-1] + 1e-12):
            raise ValueError("redshift lies outside the simulated trajectory")
        return np.exp(np.interp(z, source_z, np.log(self.f_de[::-1])))

    def to_dict(self, include_series: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed": int(self.seed),
            "dark_coefficient": float(self.dark_coefficient),
            "closure_converged": bool(self.closure_converged),
            "metadata": self.metadata,
            "summary": {
                "steps": int(self.N.size - 1),
                "z_start": float(self.redshift[0]),
                "rho_initial": float(self.rho_c[0]),
                "rho_final": float(self.rho_c[-1]),
                "max_ripple": float(np.max(self.ripple)),
                "max_abs_continuity_residual": float(
                    np.max(np.abs(self.continuity_residual))
                ),
                "max_abs_friedmann_residual": float(
                    np.max(np.abs(self.friedmann_residual))
                ),
            },
        }
        if include_series:
            payload["series"] = {
                "N": self.N.tolist(),
                "redshift": self.redshift.tolist(),
                "expansion": self.expansion.tolist(),
                "rho_c": self.rho_c.tolist(),
                "f_de": self.f_de.tolist(),
                "pressure_c": self.pressure_c.tolist(),
                "depletion_rate": self.depletion_rate.tolist(),
                "w_interval": self.w_interval.tolist(),
                "ripple": self.ripple.tolist(),
                "x_total": self.x_total.tolist(),
                "y_total": self.y_total.tolist(),
                "structure_total": self.structure_total.tolist(),
                "regime_fractions": self.regime_fractions.tolist(),
                "continuity_residual": self.continuity_residual.tolist(),
                "friedmann_residual": self.friedmann_residual.tolist(),
            }
        return payload
