"""Configuration and graph helpers for the coupled DSCD+FLRW system."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


SYSTEM_VERSION = "dscd-cosmology-v1"


@dataclass(frozen=True)
class DSCDCosmologyConfig:
    """Dimensionless late-time DSCD+FLRW realization.

    Proper time is measured in Hubble times, tau=H0*t.  The integrator uses
    N=ln(a) with d_tau=d_N/E.  Rates are therefore proper-time rates.
    """

    # Integration and background
    z_start: float = 2.6
    dN: float = 0.02
    omega_m: float = 0.30
    omega_r: float = 9.0e-5
    closure_tolerance: float = 2.0e-8
    closure_iterations: int = 12

    # Graph and initial state
    n_cells: int = 8
    initial_rho_c: float = 1.025
    initial_x: float = 4.0
    initial_y: float = 5.0
    initial_structure: float = 0.0
    initial_phase: float = 0.0

    # Asymmetric interaction dynamics
    alpha_xy: float = 0.20
    alpha_xx: float = 0.055
    omega_x: float = 1.0
    omega_y: float = 1.22
    coherence_floor: float = 0.10

    # Funded density costs per event
    cost_xy: float = 2.0e-4
    cost_xx: float = 1.0e-4
    explosion_cost: float = 1.5e-4
    bond_cost: float = 8.0e-5

    # Memory, ripple, and regimes
    ripple_threshold: float = 2.0
    ripple_margin: float = 3.0
    leakage_rate: float = 2.5e-5
    explosion_rate: float = 0.08
    bond_rate: float = 0.06
    bond_temperature: float = 0.8
    gamma_xy: float = 8.0e-4
    gamma_xx: float = 4.0e-4
    gamma_bond: float = 6.0e-4

    # Conservative transport
    diffusion_x: float = 0.04
    diffusion_y: float = 0.04

    # Reproducibility
    seed: int = 1729

    def validate(self) -> "DSCDCosmologyConfig":
        finite_values = {
            name: value
            for name, value in asdict(self).items()
            if isinstance(value, (float, int)) and name not in {"seed"}
        }
        if any(not np.isfinite(float(value)) for value in finite_values.values()):
            raise ValueError("all numeric configuration values must be finite")
        if self.z_start <= 0.0:
            raise ValueError("z_start must be positive")
        if not 0.0 < self.dN < np.log1p(self.z_start):
            raise ValueError("dN must lie between zero and the integration range")
        if self.n_cells < 1:
            raise ValueError("n_cells must be positive")
        if self.closure_iterations < 1 or self.closure_tolerance <= 0.0:
            raise ValueError("invalid closure controls")
        if self.omega_m <= 0.0 or self.omega_r < 0.0:
            raise ValueError("matter must be positive and radiation non-negative")
        if self.omega_m + self.omega_r >= 1.0:
            raise ValueError("flat background requires a positive DSCD fraction")
        if self.initial_rho_c <= 0.0:
            raise ValueError("initial_rho_c must be positive")
        if min(self.initial_x, self.initial_y, self.initial_structure) < 0.0:
            raise ValueError("initial DSCD states must be non-negative")
        if not 0.0 <= self.coherence_floor <= 1.0:
            raise ValueError("coherence_floor must lie in [0,1]")

        nonnegative = (
            "alpha_xy",
            "alpha_xx",
            "omega_x",
            "omega_y",
            "cost_xy",
            "cost_xx",
            "explosion_cost",
            "bond_cost",
            "ripple_threshold",
            "ripple_margin",
            "leakage_rate",
            "explosion_rate",
            "bond_rate",
            "bond_temperature",
            "gamma_xy",
            "gamma_xx",
            "gamma_bond",
            "diffusion_x",
            "diffusion_y",
        )
        if any(getattr(self, name) < 0.0 for name in nonnegative):
            raise ValueError("rates, costs, thresholds, and diffusion must be non-negative")
        if self.ripple_margin <= 0.0:
            raise ValueError("ripple_margin must be positive")
        return self

    @property
    def n_start(self) -> float:
        return -float(np.log1p(self.z_start))

    @property
    def omega_c_today(self) -> float:
        return 1.0 - self.omega_m - self.omega_r

    @property
    def delta_omega(self) -> float:
        return self.omega_x - self.omega_y

    def to_dict(self) -> dict[str, Any]:
        return {"system_version": SYSTEM_VERSION, **asdict(self)}

    def with_changes(self, **changes: Any) -> "DSCDCosmologyConfig":
        return replace(self, **changes).validate()


def ring_neighbors(n_cells: int) -> tuple[tuple[int, ...], ...]:
    """Return a simple undirected ring without duplicate/self edges."""
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    if n_cells == 1:
        return ((),)
    if n_cells == 2:
        return ((1,), (0,))
    return tuple(
        tuple(sorted({(index - 1) % n_cells, (index + 1) % n_cells}))
        for index in range(n_cells)
    )
