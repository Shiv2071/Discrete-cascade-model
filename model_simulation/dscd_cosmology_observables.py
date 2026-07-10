"""Observable adapters for complete DSCD+GR trajectories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from desi_bao_likelihood import (
    BAODataset,
    bao_distances_from_expansion,
    predict_dataset_from_expansion,
)
from dscd_cosmology_state import DSCDCosmologyTrajectory


def trajectory_bao_distances(
    trajectory: DSCDCosmologyTrajectory,
    redshifts: Sequence[float],
    theta: float,
    quadrature_order: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate BAO distances directly from an emergent E(z) trajectory."""
    return bao_distances_from_expansion(
        redshifts,
        theta,
        trajectory.expansion_at,
        quadrature_order=quadrature_order,
    )


def trajectory_prediction(
    trajectory: DSCDCosmologyTrajectory,
    dataset: BAODataset,
    theta: float,
    quadrature_order: int = 64,
) -> np.ndarray:
    """Predict the ordered compressed BAO vector for a dataset."""
    if float(np.max(dataset.redshifts)) > float(trajectory.redshift[0]) + 1.0e-12:
        raise ValueError("trajectory does not cover every dataset redshift")
    return predict_dataset_from_expansion(
        dataset,
        theta,
        trajectory.expansion_at,
        quadrature_order=quadrature_order,
    )


def acceleration_parameter(
    trajectory: DSCDCosmologyTrajectory,
) -> tuple[np.ndarray, np.ndarray]:
    """Return midpoint N and q=-1-d ln(H)/dN from the trajectory."""
    delta_n = np.diff(trajectory.N)
    q = -1.0 - np.diff(np.log(trajectory.expansion)) / delta_n
    midpoint = 0.5 * (trajectory.N[1:] + trajectory.N[:-1])
    return midpoint, q


def early_fraction_bound(
    trajectory: DSCDCosmologyTrajectory,
    omega_m: float,
    omega_r: float,
    redshift: float = 1100.0,
) -> float:
    """Conservative frozen-density upper bound above the simulated start.

    The initial DSCD density is held fixed while standard matter/radiation grow.
    This is an early-negligibility gate, not an early-universe DSCD simulation.
    """
    if redshift <= trajectory.redshift[0]:
        raise ValueError("early-fraction bound is only for redshifts above the run")
    dark = trajectory.dark_coefficient * trajectory.rho_c[0]
    matter = omega_m * (1.0 + redshift) ** 3
    radiation = omega_r * (1.0 + redshift) ** 4
    return float(dark / (dark + matter + radiation))
