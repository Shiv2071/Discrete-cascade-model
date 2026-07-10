"""Unit tests for the clean DESI likelihood, models, and optimizer invariants."""

from __future__ import annotations

import unittest

import numpy as np

from desi_bao_likelihood import (
    BAODataset,
    DR1,
    DR1_SOURCE,
    DR2,
    DR2_SOURCE,
    Observable,
    bao_distances,
    conditional_target_diagnostics,
)
from desi_cosmology_models import MODELS, fit_model


class OfficialDataTests(unittest.TestCase):
    def test_official_dimensions_and_qso_observable(self) -> None:
        self.assertEqual(DR1.size, 12)
        self.assertEqual(DR2.size, 13)
        qso = [item for item in DR1.observables if item.tracer == "QSO"]
        self.assertEqual([(item.redshift, item.kind) for item in qso], [(1.491, "DV")])
        self.assertAlmostEqual(DR1.values[9], 26.07217182)

    def test_pinned_source_hashes(self) -> None:
        self.assertEqual(
            DR1_SOURCE["mean_sha256"],
            "dd2873a0b88459a491af3c0c0307ba059f62df9211d5b976760f310565a1be68",
        )
        self.assertEqual(
            DR2_SOURCE["covariance_sha256"],
            "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509",
        )

    def test_covariances_are_symmetric_positive_definite(self) -> None:
        for dataset in (DR1, DR2):
            np.testing.assert_allclose(dataset.covariance, dataset.covariance.T, atol=0.0)
            self.assertTrue(np.all(np.linalg.eigvalsh(dataset.covariance) > 0.0))
            self.assertLess(np.linalg.cond(dataset.covariance), 1.0e12)

    def test_chi2_is_permutation_invariant(self) -> None:
        prediction = DR1.values + np.linspace(-0.1, 0.1, DR1.size)
        reference = DR1.chi2(prediction)
        permutation = np.asarray([9, 0, 11, 2, 7, 4, 6, 1, 10, 3, 8, 5])
        permuted = BAODataset(
            "permuted",
            tuple(DR1.observables[i] for i in permutation),
            DR1.values[permutation],
            DR1.covariance[np.ix_(permutation, permutation)],
            {},
        )
        self.assertAlmostEqual(reference, permuted.chi2(prediction[permutation]), places=10)


class DistanceAndConditionalTests(unittest.TestCase):
    def test_distance_identity_and_zero_redshift(self) -> None:
        z = np.asarray([0.0, 0.3, 1.0])
        dm, dh, dv = bao_distances(z, 0.033, 0.3, lambda x: np.ones_like(x), quadrature_order=64)
        self.assertEqual(dm[0], 0.0)
        self.assertEqual(dv[0], 0.0)
        np.testing.assert_allclose(dv, np.cbrt(z * dm * dm * dh), rtol=1.0e-13)

    def test_integration_convergence(self) -> None:
        z = np.asarray([0.295, 0.93, 2.33])
        low = bao_distances(z, 0.033, 0.31, lambda x: np.ones_like(x), quadrature_order=48)
        high = bao_distances(z, 0.033, 0.31, lambda x: np.ones_like(x), quadrature_order=96)
        for low_component, high_component in zip(low, high):
            np.testing.assert_allclose(low_component, high_component, rtol=2.0e-12, atol=1.0e-12)

    def test_invalid_cosmologies_fail(self) -> None:
        with self.assertRaises(ValueError):
            bao_distances([1.0], 0.0, 0.3, lambda x: np.ones_like(x))
        with self.assertRaises(ValueError):
            bao_distances([1.0], 0.03, 1.1, lambda x: np.ones_like(x))
        with self.assertRaises(ValueError):
            bao_distances([1.0], 0.03, 0.3, lambda x: -np.ones_like(x))

    def test_conditional_covariance_schur_complement(self) -> None:
        covariance = np.asarray([[1.0, 0.4], [0.4, 2.0]])
        toy = BAODataset(
            "toy",
            (Observable(0.2, "DV", "a"), Observable(1.0, "DV", "b")),
            np.asarray([1.0, 2.0]),
            covariance,
            {},
        )
        result = conditional_target_diagnostics(toy, [1.5, 3.0], [0], [1])
        expected_residual = 1.0 - 0.4 * 0.5
        expected_variance = 2.0 - 0.4**2
        self.assertAlmostEqual(result["conditional_residuals"][0], expected_residual)
        self.assertAlmostEqual(result["conditional_covariance"][0][0], expected_variance)
        self.assertAlmostEqual(result["chi2"], expected_residual**2 / expected_variance)

    def test_zero_cross_block_reduces_to_target_likelihood(self) -> None:
        prediction = MODELS["LCDM"].predict(DR1, [0.033, 0.3])
        train = list(range(7))
        target = list(range(7, 12))
        conditional = conditional_target_diagnostics(DR1, prediction, train, target)
        marginal = DR1.subset(target).chi2(prediction[target])
        self.assertEqual(conditional["cross_covariance_max_abs"], 0.0)
        self.assertAlmostEqual(conditional["chi2"], marginal, places=10)


class ModelAndOptimizerTests(unittest.TestCase):
    def test_lambda_nesting(self) -> None:
        lambda_prediction = MODELS["LCDM"].predict(DR2, [0.033, 0.3])
        for name, parameters in (
            ("WCDM_NONPHANTOM", [0.033, 0.3, -1.0]),
            ("PHEN_LOG", [0.033, 0.3, -1.0]),
            ("PHEN_POLY", [0.033, 0.3, -1.0, 7.0]),
            ("CPL", [0.033, 0.3, -1.0, 0.0]),
            ("CPL_NONPHANTOM", [0.033, 0.3, -1.0, 0.0]),
            ("MONO_ONE", [0.033, 0.3, 0.0]),
        ):
            np.testing.assert_allclose(
                MODELS[name].predict(DR2, parameters),
                lambda_prediction,
                rtol=1.0e-13,
                atol=1.0e-13,
            )

    def test_nonphantom_cpl_constraint(self) -> None:
        with self.assertRaises(ValueError):
            MODELS["CPL_NONPHANTOM"].predict(DR1, [0.033, 0.3, -0.9, -0.2])
        MODELS["CPL_NONPHANTOM"].predict(DR1, [0.033, 0.3, -0.9, -0.1])

    def test_phen_poly_gamma_is_unidentifiable_at_lambda_boundary(self) -> None:
        first = MODELS["PHEN_POLY"].predict(DR1, [0.033, 0.3, -1.0, 0.0])
        second = MODELS["PHEN_POLY"].predict(DR1, [0.033, 0.3, -1.0, 25.0])
        np.testing.assert_allclose(first, second, rtol=1.0e-13, atol=1.0e-13)

    def test_optimizer_recovers_synthetic_lcdm_deterministically(self) -> None:
        truth = np.asarray([0.0332, 0.31])
        values = MODELS["LCDM"].predict(DR1, truth)
        synthetic = BAODataset(
            "synthetic",
            DR1.observables,
            values,
            DR1.covariance,
            {"test": True},
        )
        first = fit_model(MODELS["LCDM"], synthetic, seeds=[177], maxiter=45)
        second = fit_model(MODELS["LCDM"], synthetic, seeds=[177], maxiter=45)
        np.testing.assert_allclose(first.parameters, truth, rtol=2.0e-4, atol=2.0e-5)
        np.testing.assert_allclose(first.parameters, second.parameters, rtol=0.0, atol=1.0e-11)
        self.assertLess(first.chi2, 1.0e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
