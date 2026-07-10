"""Unit and integration tests for the coupled DSCD+FLRW system."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from desi_bao_likelihood import bao_distances, bao_distances_from_expansion
from dscd_cosmology_config import DSCDCosmologyConfig, ring_neighbors
from dscd_cosmology_dynamics import (
    DSCDCosmologySystem,
    fund_count,
    interval_equation_of_state,
    pressure_from_depletion,
)
from dscd_cosmology_ensemble import run_ensemble
from dscd_cosmology_observables import early_fraction_bound


class FundingAndConfigurationTests(unittest.TestCase):
    def test_funding_never_overdraws(self) -> None:
        count, budget = fund_count(10, 8, 0.3, 1.0)
        self.assertEqual(count, 3)
        self.assertGreaterEqual(budget, 0.0)
        self.assertAlmostEqual(budget, 0.1)

    def test_ring_graph_is_undirected(self) -> None:
        graph = ring_neighbors(7)
        for index, row in enumerate(graph):
            for neighbor in row:
                self.assertIn(index, graph[neighbor])

    def test_invalid_flat_background_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DSCDCosmologyConfig(omega_m=1.0).validate()

    def test_pressure_and_interval_w_are_nonphantom_for_depletion(self) -> None:
        pressure = pressure_from_depletion(1.0, 0.03, 1.2)
        self.assertAlmostEqual(-0.03 + 3.0 * 1.2 * (1.0 + pressure), 0.0)
        w = interval_equation_of_state(1.0, 0.999, 0.02)
        self.assertGreater(w, -1.0)


class CoupledTrajectoryTests(unittest.TestCase):
    def test_zero_activity_reproduces_lambda_background(self) -> None:
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
        trajectory = DSCDCosmologySystem(config).simulate(11)
        expected = np.sqrt(
            config.omega_m * np.exp(-3.0 * trajectory.N)
            + config.omega_r * np.exp(-4.0 * trajectory.N)
            + config.omega_c_today
        )
        np.testing.assert_allclose(trajectory.expansion, expected, atol=2e-7)
        self.assertTrue(trajectory.closure_converged)

    def test_active_path_is_funded_and_conservative(self) -> None:
        config = DSCDCosmologyConfig(z_start=0.8, dN=0.02).validate()
        trajectory = DSCDCosmologySystem(config).simulate(13)
        self.assertTrue(np.all(np.diff(trajectory.rho_c) <= 1.0e-14))
        self.assertLess(np.max(np.abs(trajectory.continuity_residual)), 1.0e-10)
        self.assertLess(np.max(np.abs(trajectory.friedmann_residual)), 1.0e-10)
        self.assertTrue(np.all(trajectory.w_interval >= -1.0 - 1.0e-12))

    def test_trajectory_distance_interface_matches_lcdm_utility(self) -> None:
        config = DSCDCosmologyConfig(
            z_start=2.5,
            dN=0.005,
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
        trajectory = DSCDCosmologySystem(config).simulate(19)
        z = np.asarray([0.3, 0.8, 2.0])
        theta = 0.033
        direct = bao_distances_from_expansion(z, theta, trajectory.expansion_at, 64)
        reference = bao_distances(
            z,
            theta,
            config.omega_m,
            lambda value: np.ones_like(value),
            omega_r=config.omega_r,
            quadrature_order=64,
        )
        for observed, expected in zip(direct, reference):
            np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-7)

    def test_ensemble_seed_manifest_is_reproducible(self) -> None:
        config = DSCDCosmologyConfig(z_start=0.5, dN=0.03, n_cells=2).validate()
        first = run_ensemble(config, [3, 5], variance_semantics="monte_carlo")
        second = run_ensemble(config, [3, 5], variance_semantics="monte_carlo")
        np.testing.assert_array_equal(
            first.trajectories[0].rho_c, second.trajectories[0].rho_c
        )

    def test_early_density_is_negligible_under_late_time_boundary(self) -> None:
        config = DSCDCosmologyConfig(z_start=2.6, dN=0.03).validate()
        trajectory = DSCDCosmologySystem(config).simulate(23)
        fraction = early_fraction_bound(
            trajectory, config.omega_m, config.omega_r, redshift=1100.0
        )
        self.assertLess(fraction, 1.0e-8)

    def test_zero_beat_ablation_is_a_distinct_configuration(self) -> None:
        config = DSCDCosmologyConfig().validate()
        ablation = replace(config, omega_y=config.omega_x).validate()
        self.assertNotEqual(config.delta_omega, ablation.delta_omega)
        self.assertEqual(ablation.delta_omega, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
