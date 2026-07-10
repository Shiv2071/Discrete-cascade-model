"""Unit tests for the v2 history-compatible DSCD forecast components."""

from __future__ import annotations

import unittest

import numpy as np

from desi_bao_likelihood import DR1, DR2
from dscd_forecast_prior import (
    PRIOR_LOWER,
    PRIOR_NAMES,
    PRIOR_UPPER,
    PriorSample,
    effective_sample_size,
    sample_prior,
    trajectory_seeds,
    weighted_quantiles,
)
from dscd_history_ensemble import (
    evaluate_sample,
    importance_weights,
    profile_theta,
    with_values,
)


class PriorTests(unittest.TestCase):
    def test_samples_respect_declared_support(self) -> None:
        samples = sample_prior(64, 11)
        matrix = np.stack([item.values for item in samples])
        self.assertEqual(matrix.shape, (64, len(PRIOR_NAMES)))
        self.assertTrue(np.all(matrix >= PRIOR_LOWER - 1e-12))
        self.assertTrue(np.all(matrix <= PRIOR_UPPER + 1e-12))

    def test_sampling_is_seed_reproducible(self) -> None:
        first = np.stack([item.values for item in sample_prior(16, 99)])
        second = np.stack([item.values for item in sample_prior(16, 99)])
        np.testing.assert_array_equal(first, second)

    def test_out_of_support_sample_rejected(self) -> None:
        bad = PRIOR_UPPER.copy()
        bad[0] += 1.0
        with self.assertRaises(ValueError):
            PriorSample(0, bad)

    def test_config_mapping_applies_depletion_scale(self) -> None:
        values = 0.5 * (PRIOR_LOWER + PRIOR_UPPER)
        values[PRIOR_NAMES.index("log10_depletion_scale")] = 2.0
        sample = PriorSample(3, values)
        config = sample.to_config()
        from dscd_cosmology_config import DSCDCosmologyConfig

        base = DSCDCosmologyConfig()
        self.assertAlmostEqual(config.cost_xy, base.cost_xy * 100.0)
        self.assertAlmostEqual(config.leakage_rate, base.leakage_rate * 100.0)
        self.assertAlmostEqual(
            config.omega_y - config.omega_x,
            float(values[PRIOR_NAMES.index("delta_omega")]),
        )
        self.assertAlmostEqual(
            config.initial_y / config.initial_x,
            float(values[PRIOR_NAMES.index("asymmetry_ratio")]),
        )

    def test_trajectory_seeds_unique_and_group_disjoint(self) -> None:
        group_a = {
            seed
            for index in range(200)
            for seed in trajectory_seeds(index, 4, group_offset=0)
        }
        group_b = {
            seed
            for index in range(200)
            for seed in trajectory_seeds(index, 4, group_offset=1)
        }
        self.assertEqual(len(group_a), 800)
        self.assertEqual(len(group_b), 800)
        self.assertFalse(group_a & group_b)


class WeightingTests(unittest.TestCase):
    def test_weighted_quantiles_match_unweighted_median(self) -> None:
        values = np.asarray([4.0, 1.0, 3.0, 2.0, 5.0])
        weights = np.ones(5)
        result = weighted_quantiles(values, weights, [0.5])
        self.assertAlmostEqual(float(result[0]), 3.0)

    def test_weighted_quantiles_follow_weight_mass(self) -> None:
        values = np.asarray([0.0, 10.0])
        weights = np.asarray([1.0, 99.0])
        result = weighted_quantiles(values, weights, [0.5])
        self.assertGreater(float(result[0]), 9.0)

    def test_effective_sample_size_limits(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.ones(10) / 10.0), 10.0)
        one_hot = np.zeros(10)
        one_hot[3] = 1.0
        self.assertAlmostEqual(effective_sample_size(one_hot), 1.0)

    def test_importance_weights_softmax_and_failures(self) -> None:
        records = [
            {"status": "OK", "joint_log_density": -1.0},
            {"status": "OK", "joint_log_density": -3.0},
            {"status": "FAILED"},
        ]
        weights = importance_weights(records)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertEqual(float(weights[2]), 0.0)
        self.assertAlmostEqual(weights[0] / weights[1], np.exp(2.0), places=10)

    def test_importance_weights_require_a_valid_sample(self) -> None:
        with self.assertRaises(ValueError):
            importance_weights([{"status": "FAILED"}])


class ThetaProfilingTests(unittest.TestCase):
    def test_profile_theta_recovers_exact_scaling(self) -> None:
        truth_theta = 0.0334
        unit_mean = DR2.values * truth_theta
        synthetic = with_values(DR2, DR2.values, "_test")
        theta, clipped = profile_theta([unit_mean], [synthetic])
        self.assertFalse(clipped)
        self.assertAlmostEqual(theta, truth_theta, places=12)

    def test_profile_theta_reports_clipping(self) -> None:
        unit_mean = DR2.values * 0.5
        theta, clipped = profile_theta([unit_mean], [DR2])
        self.assertTrue(clipped)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_sample_produces_complete_record(self) -> None:
        sample = sample_prior(2, 12345)[0]
        record = evaluate_sample(sample, replicates=2)
        self.assertEqual(record["status"], "OK")
        self.assertIn("joint_log_density", record)
        self.assertEqual(record["dr3_release"], DR2.name)
        block = np.asarray(record["dr3_predictions"])
        self.assertEqual(block.ndim, 3)
        self.assertEqual(block.shape[0], 2)
        self.assertEqual(block.shape[2], DR2.size)
        scale_weights = np.asarray(record["dr3_scale_weights"])
        self.assertEqual(scale_weights.size, block.shape[1])
        self.assertAlmostEqual(float(np.sum(scale_weights)), 1.0)
        self.assertGreater(record["scale_sigma"], 0.0)
        self.assertEqual(len(record["per_release"]), 2)
        self.assertEqual(
            len(record["w_interval_mean"]),
            len(record["f_de_mean"]) - 1,
        )

    def test_evaluate_sample_scores_both_official_releases(self) -> None:
        sample = sample_prior(2, 12345)[1]
        record = evaluate_sample(sample, replicates=2)
        self.assertEqual(record["status"], "OK")
        self.assertEqual(set(record["per_release"]), {DR1.name, DR2.name})
        for release in record["per_release"].values():
            self.assertTrue(np.isfinite(release["log_density"]))
            self.assertGreaterEqual(release["chi2_observational"], 0.0)


if __name__ == "__main__":
    unittest.main()
