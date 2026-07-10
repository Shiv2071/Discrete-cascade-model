"""Prior over latent DSCD cosmological realizations for the v2 forecast.

Version 2 treats the present DSCD state and its structural dynamics as
latent.  Instead of freezing one hand-picked configuration (version 1),
a declared prior over depletion strength, interaction structure, beat
asymmetry, initial activity, and the matter fraction is sampled with a
scrambled Sobol sequence.  Every sample is one complete coupled DSCD+GR
realization; history compatibility is decided by data, not by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.stats import qmc

from dscd_cosmology_config import DSCDCosmologyConfig
from dscd_cosmology_inference import apply_depletion_scale


FORECAST_VERSION = "dscd-forecast-v2"

# Each entry: (name, lower, upper, description).  Bounds are declared once
# here and never tuned against DESI data.
PRIOR_DIMENSIONS: tuple[tuple[str, float, float, str], ...] = (
    (
        "log10_depletion_scale",
        -1.0,
        3.0,
        "log10 multiplier on every funded density cost and leakage rate",
    ),
    ("alpha_xy", 0.05, 0.60, "cross-species interaction strength"),
    ("alpha_xx", 0.00, 0.20, "same-species depletion strength"),
    ("delta_omega", 0.00, 0.80, "beat asymmetry omega_y - omega_x"),
    ("initial_x", 2.0, 10.0, "initial X activity per cell"),
    ("asymmetry_ratio", 1.0, 2.0, "initial Y/X abundance ratio"),
    ("explosion_rate", 0.0, 0.20, "explosive regime event rate"),
    ("omega_m", 0.20, 0.45, "present matter fraction"),
)
PRIOR_NAMES = tuple(item[0] for item in PRIOR_DIMENSIONS)
PRIOR_LOWER = np.asarray([item[1] for item in PRIOR_DIMENSIONS])
PRIOR_UPPER = np.asarray([item[2] for item in PRIOR_DIMENSIONS])

THETA_BOUNDS = (0.020, 0.045)


@dataclass(frozen=True)
class PriorSample:
    """One latent realization drawn from the declared v2 prior."""

    index: int
    values: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=float)
        if array.shape != (len(PRIOR_NAMES),) or np.any(~np.isfinite(array)):
            raise ValueError("prior sample has the wrong shape or non-finite values")
        if np.any(array < PRIOR_LOWER - 1e-12) or np.any(array > PRIOR_UPPER + 1e-12):
            raise ValueError("prior sample lies outside the declared support")
        object.__setattr__(self, "values", array)

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(PRIOR_NAMES, self.values)}

    @property
    def depletion_scale(self) -> float:
        return float(10.0 ** self.values[PRIOR_NAMES.index("log10_depletion_scale")])

    def to_config(self, base: DSCDCosmologyConfig | None = None) -> DSCDCosmologyConfig:
        """Map the latent sample to one validated coupled-system configuration."""
        parameters = self.as_dict()
        template = base if base is not None else DSCDCosmologyConfig()
        configured = template.with_changes(
            alpha_xy=parameters["alpha_xy"],
            alpha_xx=parameters["alpha_xx"],
            omega_y=template.omega_x + parameters["delta_omega"],
            initial_x=parameters["initial_x"],
            initial_y=parameters["initial_x"] * parameters["asymmetry_ratio"],
            explosion_rate=parameters["explosion_rate"],
            omega_m=parameters["omega_m"],
        )
        return apply_depletion_scale(configured, self.depletion_scale)


def prior_metadata() -> dict[str, Any]:
    return {
        "forecast_version": FORECAST_VERSION,
        "dimensions": [
            {"name": name, "lower": lower, "upper": upper, "description": text}
            for name, lower, upper, text in PRIOR_DIMENSIONS
        ],
        "theta_bounds": list(THETA_BOUNDS),
        "theta_treatment": "profiled analytically per sample (exact 1/theta scaling)",
        "sampler": "scrambled Sobol (scipy.stats.qmc.Sobol)",
    }


def sample_prior(count: int, sampler_seed: int) -> tuple[PriorSample, ...]:
    """Draw `count` low-discrepancy samples from the declared prior.

    The first half of a Sobol sequence is itself low-discrepancy, so
    nested convergence checks may compare the first count//2 samples
    against the full set.
    """
    if count < 2:
        raise ValueError("at least two prior samples are required")
    engine = qmc.Sobol(d=len(PRIOR_NAMES), scramble=True, seed=int(sampler_seed))
    unit = engine.random(int(count))
    scaled = qmc.scale(unit, PRIOR_LOWER, PRIOR_UPPER)
    return tuple(PriorSample(index, row) for index, row in enumerate(scaled))


def trajectory_seeds(
    sample_index: int, replicates: int, *, group_offset: int = 0
) -> tuple[int, ...]:
    """Deterministic, collision-free per-sample stochastic seeds.

    `group_offset` selects a disjoint seed universe (used by the seed
    robustness gate); indices within one group never collide because each
    sample owns a contiguous block of 64 seeds.
    """
    if replicates < 1 or replicates > 64:
        raise ValueError("replicates must lie in [1, 64]")
    base = 100_000 + int(group_offset) * 10_000_000 + int(sample_index) * 64
    return tuple(base + j for j in range(replicates))


def weighted_quantiles(
    values: Sequence[float],
    weights: Sequence[float],
    quantiles: Sequence[float],
) -> np.ndarray:
    """Weighted empirical quantiles with cumulative-weight interpolation."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if v.shape != w.shape or v.ndim != 1 or v.size == 0:
        raise ValueError("values and weights must be equal-length vectors")
    if np.any(w < 0.0) or not np.isfinite(np.sum(w)) or np.sum(w) <= 0.0:
        raise ValueError("weights must be non-negative with positive finite sum")
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted) - 0.5 * w_sorted
    cumulative /= np.sum(w_sorted)
    return np.interp(q, cumulative, v_sorted)


def effective_sample_size(weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    return float(total**2 / np.sum(w**2))
