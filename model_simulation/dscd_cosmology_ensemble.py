"""Reproducible stochastic ensembles for cosmological DSCD trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from desi_bao_likelihood import BAODataset
from dscd_cosmology_config import DSCDCosmologyConfig
from dscd_cosmology_dynamics import DSCDCosmologySystem
from dscd_cosmology_observables import trajectory_prediction
from dscd_cosmology_state import DSCDCosmologyTrajectory


@dataclass(frozen=True)
class DSCDEnsembleResult:
    """A fixed-seed ensemble; trajectories remain the scientific primitive."""

    trajectories: tuple[DSCDCosmologyTrajectory, ...]
    seeds: tuple[int, ...]
    variance_semantics: str

    def __post_init__(self) -> None:
        if not self.trajectories or len(self.trajectories) != len(self.seeds):
            raise ValueError("ensemble requires one trajectory per seed")
        reference = self.trajectories[0].N
        for trajectory in self.trajectories[1:]:
            np.testing.assert_allclose(trajectory.N, reference, rtol=0.0, atol=1.0e-14)
        if self.variance_semantics not in {"physical", "monte_carlo"}:
            raise ValueError("variance_semantics must be physical or monte_carlo")

    @property
    def N(self) -> np.ndarray:
        return self.trajectories[0].N

    def stack(self, field: str) -> np.ndarray:
        return np.stack([np.asarray(getattr(item, field)) for item in self.trajectories])

    def summary(self) -> dict[str, Any]:
        expansion = self.stack("expansion")
        density = self.stack("rho_c")
        return {
            "replicates": len(self.trajectories),
            "seeds": list(self.seeds),
            "variance_semantics": self.variance_semantics,
            "all_closure_converged": all(
                item.closure_converged for item in self.trajectories
            ),
            "expansion_median": np.median(expansion, axis=0).tolist(),
            "expansion_p16": np.percentile(expansion, 16.0, axis=0).tolist(),
            "expansion_p84": np.percentile(expansion, 84.0, axis=0).tolist(),
            "rho_c_median": np.median(density, axis=0).tolist(),
            "rho_c_p16": np.percentile(density, 16.0, axis=0).tolist(),
            "rho_c_p84": np.percentile(density, 84.0, axis=0).tolist(),
        }

    def predictive_samples(
        self,
        dataset: BAODataset,
        theta: float,
        quadrature_order: int = 64,
    ) -> np.ndarray:
        return np.stack(
            [
                trajectory_prediction(item, dataset, theta, quadrature_order)
                for item in self.trajectories
            ]
        )


def run_ensemble(
    config: DSCDCosmologyConfig,
    seeds: Sequence[int],
    *,
    variance_semantics: str = "physical",
) -> DSCDEnsembleResult:
    selected = tuple(int(seed) for seed in seeds)
    if len(set(selected)) != len(selected):
        raise ValueError("ensemble seeds must be unique")
    system = DSCDCosmologySystem(config)
    trajectories = tuple(system.simulate(seed) for seed in selected)
    return DSCDEnsembleResult(trajectories, selected, variance_semantics)
