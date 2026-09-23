"""Tests for `b2t_drift.analysis.subspace_alignment` -- Phase 3's CORAL and
orthogonal Procrustes unsupervised alignment (Arm 1a/1b)."""

from __future__ import annotations

import numpy as np
import pytest

from b2t_drift.analysis.subspace_alignment import (
    CovarianceAccumulator,
    apply_alignment,
    coral_alignment_matrix,
    procrustes_alignment_matrix,
    shrink_covariance,
)
from b2t_drift.analysis.subspace_alignment import _symmetric_matrix_power  # noqa: PLC2701 -- direct unit test of the shared helper


def _random_spd(p: int, rng: np.random.Generator, condition: float = 10.0) -> np.ndarray:
    """A random symmetric positive-definite `(p, p)` matrix with a roughly
    controlled condition number, for reproducible synthetic tests."""
    q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    eigenvalues = np.linspace(1.0, condition, p)
    return (q * eigenvalues) @ q.T


def _sample_from_covariance(cov: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean samples whose EMPIRICAL covariance approaches `cov` as
    `n_samples` grows (via a Cholesky-based linear transform of iid
    standard normal noise)."""
    p = cov.shape[0]
    l = np.linalg.cholesky(cov + 1e-9 * np.eye(p))
    z = rng.normal(size=(n_samples, p))
    return z @ l.T


class TestCovarianceAccumulator:
    def test_matches_numpy_cov_in_one_shot(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 8))
        acc = CovarianceAccumulator(n_features=8)
        acc.add(x)
        expected = np.cov(x, rowvar=False, bias=True)
        np.testing.assert_allclose(acc.covariance(), expected, atol=1e-8)
        np.testing.assert_allclose(acc.mean(), x.mean(axis=0), atol=1e-8)
        assert acc.n_samples == 500

    def test_matches_numpy_cov_across_multiple_add_calls(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 6))
        acc = CovarianceAccumulator(n_features=6)
        for chunk in np.array_split(x, 7):  # simulates one add() per trial
            acc.add(chunk)
        expected = np.cov(x, rowvar=False, bias=True)
        np.testing.assert_allclose(acc.covariance(), expected, atol=1e-8)

    def test_wrong_feature_width_raises(self):
        acc = CovarianceAccumulator(n_features=8)
        with pytest.raises(ValueError, match="expected 8 features"):
            acc.add(np.zeros((10, 4)))

    def test_empty_accumulator_raises(self):
        acc = CovarianceAccumulator(n_features=4)
        with pytest.raises(ValueError, match="no samples"):
            acc.mean()
        with pytest.raises(ValueError, match="no samples"):
            acc.covariance()


class TestShrinkCovariance:
    def test_alpha_zero_is_unchanged(self):
        rng = np.random.default_rng(2)
        cov = _random_spd(5, rng)
        np.testing.assert_allclose(shrink_covariance(cov, 0.0), cov)

    def test_alpha_one_is_isotropic_with_same_trace(self):
        rng = np.random.default_rng(3)
        cov = _random_spd(5, rng)
        shrunk = shrink_covariance(cov, 1.0)
        expected_diag = np.trace(cov) / 5
        np.testing.assert_allclose(shrunk, expected_diag * np.eye(5), atol=1e-10)
        assert np.trace(shrunk) == pytest.approx(np.trace(cov))

    def test_out_of_range_alpha_raises(self):
        cov = np.eye(3)
        with pytest.raises(ValueError, match="alpha"):
            shrink_covariance(cov, 1.5)
        with pytest.raises(ValueError, match="alpha"):
            shrink_covariance(cov, -0.1)


class TestSymmetricMatrixPower:
    def test_sqrt_squared_recovers_original(self):
        rng = np.random.default_rng(4)
        cov = _random_spd(6, rng)
        result = _symmetric_matrix_power(cov, eps=0.0)
        np.testing.assert_allclose(result.sqrt @ result.sqrt, cov, atol=1e-6)

    def test_sqrt_and_inv_sqrt_are_inverses(self):
        rng = np.random.default_rng(5)
        cov = _random_spd(6, rng)
        result = _symmetric_matrix_power(cov, eps=1e-8)
        np.testing.assert_allclose(result.sqrt @ result.inv_sqrt, np.eye(6), atol=1e-5)

    def test_condition_number_matches_eigenvalue_ratio(self):
        rng = np.random.default_rng(6)
        cov = _random_spd(6, rng, condition=25.0)
        result = _symmetric_matrix_power(cov, eps=1e-8)
        expected = np.linalg.cond(cov)
        assert result.condition_number == pytest.approx(expected, rel=1e-3)


class TestCoralAlignment:
    def test_identity_when_source_equals_target(self):
        rng = np.random.default_rng(7)
        cov = _random_spd(6, rng)
        result = coral_alignment_matrix(cov, cov, eps=1e-10)
        np.testing.assert_allclose(result.transform, np.eye(6), atol=1e-3)

    def test_recovers_source_covariance_on_synthetic_data(self):
        rng = np.random.default_rng(8)
        source_cov = _random_spd(5, rng, condition=8.0)
        target_cov = _random_spd(5, rng, condition=20.0)
        target_data = _sample_from_covariance(target_cov, n_samples=200_000, rng=rng)

        result = coral_alignment_matrix(source_cov, np.cov(target_data, rowvar=False, bias=True), eps=1e-8)
        aligned = apply_alignment(target_data, result.transform)
        aligned_cov = np.cov(aligned, rowvar=False, bias=True)
        np.testing.assert_allclose(aligned_cov, source_cov, atol=0.05)

    def test_reports_condition_numbers(self):
        rng = np.random.default_rng(9)
        source_cov = _random_spd(5, rng, condition=5.0)
        target_cov = _random_spd(5, rng, condition=500.0)
        result = coral_alignment_matrix(source_cov, target_cov, eps=1e-8)
        assert result.target_condition_number == pytest.approx(np.linalg.cond(target_cov), rel=1e-2)
        assert result.source_condition_number == pytest.approx(np.linalg.cond(source_cov), rel=1e-2)

    def test_shrinkage_reduces_target_condition_number(self):
        rng = np.random.default_rng(10)
        source_cov = _random_spd(5, rng, condition=5.0)
        target_cov = _random_spd(5, rng, condition=1000.0)
        result = coral_alignment_matrix(source_cov, target_cov, eps=1e-8, shrinkage_alpha=0.5)
        assert result.target_condition_number_after_shrinkage < result.target_condition_number
        assert result.shrinkage_alpha == 0.5


class TestProcrustesAlignment:
    def test_transform_is_orthogonal(self):
        rng = np.random.default_rng(11)
        source_cov = _random_spd(6, rng, condition=8.0)
        target_cov = _random_spd(6, rng, condition=30.0)
        result = procrustes_alignment_matrix(source_cov, target_cov, eps=1e-8)
        np.testing.assert_allclose(result.transform @ result.transform.T, np.eye(6), atol=1e-6)

    def test_identity_when_source_equals_target(self):
        rng = np.random.default_rng(12)
        cov = _random_spd(6, rng)
        result = procrustes_alignment_matrix(cov, cov, eps=1e-10)
        np.testing.assert_allclose(result.transform, np.eye(6), atol=1e-3)

    def test_never_rescales_variance(self):
        """Procrustes can only rotate -- applying it to data drawn from
        target_cov must leave the aligned data's TOTAL variance (trace of
        covariance) unchanged, unlike CORAL."""
        rng = np.random.default_rng(13)
        source_cov = _random_spd(5, rng, condition=8.0)
        target_cov = _random_spd(5, rng, condition=20.0)
        target_data = _sample_from_covariance(target_cov, n_samples=50_000, rng=rng)
        empirical_target_cov = np.cov(target_data, rowvar=False, bias=True)

        result = procrustes_alignment_matrix(source_cov, empirical_target_cov, eps=1e-8)
        aligned = apply_alignment(target_data, result.transform)
        aligned_cov = np.cov(aligned, rowvar=False, bias=True)
        assert np.trace(aligned_cov) == pytest.approx(np.trace(empirical_target_cov), rel=1e-2)

    def test_shrinkage_reduces_target_condition_number(self):
        rng = np.random.default_rng(14)
        source_cov = _random_spd(5, rng, condition=5.0)
        target_cov = _random_spd(5, rng, condition=1000.0)
        result = procrustes_alignment_matrix(source_cov, target_cov, eps=1e-8, shrinkage_alpha=0.5)
        assert result.target_condition_number_after_shrinkage < result.target_condition_number


class TestApplyAlignment:
    def test_shape_and_dtype_preserved(self):
        rng = np.random.default_rng(15)
        features = rng.normal(size=(10, 20, 5)).astype(np.float32)
        transform = np.eye(5, dtype=np.float64)
        aligned = apply_alignment(features, transform)
        assert aligned.shape == features.shape
        assert aligned.dtype == features.dtype

    def test_matches_manual_matmul(self):
        rng = np.random.default_rng(16)
        features = rng.normal(size=(30, 4))
        transform = rng.normal(size=(4, 4))
        np.testing.assert_allclose(apply_alignment(features, transform), features @ transform)
