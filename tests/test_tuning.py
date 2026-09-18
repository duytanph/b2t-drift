"""Tests for b2t_drift.analysis.tuning (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from b2t_drift.analysis.tuning import (
    TuningMatrixAccumulator,
    flattened_matrix_correlation,
    per_phoneme_correlation,
)


def test_single_trial_uniform_posterior_gives_mean_feature_per_class() -> None:
    """If every bin has posterior 1.0 on a single class, that class's row
    is exactly the mean feature vector over those bins -- the simplest
    sanity check of the weighted-average machinery."""
    acc = TuningMatrixAccumulator(n_features=4, n_classes=3)
    features = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    posteriors = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])  # both bins: class 1

    acc.add_trial(posteriors, features)
    matrix, weights = acc.matrix()

    assert weights[1] == pytest.approx(2.0)
    assert matrix[1] == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert np.isnan(matrix[0]).all()  # class 0 never received any weight
    assert np.isnan(matrix[2]).all()


def test_fractional_posterior_weights_contribution_correctly() -> None:
    """A bin with 0.6 probability of class 1 contributes 0.6 of its
    feature vector to class 1's sum, not the whole vector (hard
    assignment would be wrong) -- the module docstring's own example."""
    acc = TuningMatrixAccumulator(n_features=2, n_classes=2)
    features = np.array([[10.0, 20.0]])
    posteriors = np.array([[0.4, 0.6]])

    acc.add_trial(posteriors, features)
    matrix, weights = acc.matrix()

    assert weights[0] == pytest.approx(0.4)
    assert weights[1] == pytest.approx(0.6)
    # matrix[c] = sum(posterior*feature) / sum(posterior) = feature itself
    # here (only one bin), for BOTH classes -- weighting doesn't change the
    # per-class MEAN when there's only one contributing bin, only the
    # WEIGHT recorded for it.
    assert matrix[0] == pytest.approx([10.0, 20.0])
    assert matrix[1] == pytest.approx([10.0, 20.0])


def test_accumulates_correctly_across_multiple_trials() -> None:
    acc = TuningMatrixAccumulator(n_features=1, n_classes=2)
    acc.add_trial(np.array([[1.0, 0.0]]), np.array([[10.0]]))
    acc.add_trial(np.array([[1.0, 0.0]]), np.array([[20.0]]))
    acc.add_trial(np.array([[0.0, 1.0]]), np.array([[100.0]]))

    matrix, weights = acc.matrix()
    assert weights[0] == pytest.approx(2.0)
    assert weights[1] == pytest.approx(1.0)
    assert matrix[0, 0] == pytest.approx(15.0)  # mean of 10, 20
    assert matrix[1, 0] == pytest.approx(100.0)


def test_add_trial_raises_on_timestep_mismatch() -> None:
    acc = TuningMatrixAccumulator(n_features=2, n_classes=2)
    with pytest.raises(ValueError, match="timesteps"):
        acc.add_trial(np.zeros((5, 2)), np.zeros((4, 2)))


def test_add_trial_raises_on_class_count_mismatch() -> None:
    acc = TuningMatrixAccumulator(n_features=2, n_classes=3)
    with pytest.raises(ValueError, match="classes"):
        acc.add_trial(np.zeros((5, 2)), np.zeros((5, 2)))


def test_add_trial_raises_on_feature_count_mismatch() -> None:
    acc = TuningMatrixAccumulator(n_features=4, n_classes=2)
    with pytest.raises(ValueError, match="columns"):
        acc.add_trial(np.zeros((5, 2)), np.zeros((5, 2)))


def test_per_phoneme_correlation_identical_matrices_gives_one() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(5, 10))
    result = per_phoneme_correlation(matrix, matrix.copy())
    assert result["mean_correlation"] == pytest.approx(1.0, abs=1e-8)
    assert result["n_valid_classes"] == 5
    for r in result["per_class"].values():
        assert r == pytest.approx(1.0, abs=1e-8)


def test_per_phoneme_correlation_detects_reduced_similarity() -> None:
    rng = np.random.default_rng(1)
    matrix_a = rng.normal(size=(3, 20))
    matrix_b = -matrix_a  # perfectly ANTI-correlated per row
    result = per_phoneme_correlation(matrix_a, matrix_b)
    assert result["mean_correlation"] == pytest.approx(-1.0, abs=1e-8)


def test_per_phoneme_correlation_skips_nan_and_constant_rows() -> None:
    matrix_a = np.array([
        [1.0, 2.0, 3.0],       # normal row
        [np.nan, np.nan, np.nan],  # unsupported class (no weight)
        [5.0, 5.0, 5.0],       # constant row -- undefined correlation
    ])
    matrix_b = np.array([
        [2.0, 4.0, 6.0],
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
    ])
    result = per_phoneme_correlation(matrix_a, matrix_b)
    assert result["n_valid_classes"] == 1
    assert result["per_class"][0] == pytest.approx(1.0)
    assert np.isnan(result["per_class"][1])
    assert np.isnan(result["per_class"][2])
    assert result["mean_correlation"] == pytest.approx(1.0)


def test_per_phoneme_correlation_respects_class_indices_subset() -> None:
    rng = np.random.default_rng(2)
    matrix_a = rng.normal(size=(5, 10))
    matrix_b = rng.normal(size=(5, 10))
    result_all = per_phoneme_correlation(matrix_a, matrix_b)
    result_subset = per_phoneme_correlation(matrix_a, matrix_b, class_indices=[1, 3])
    assert set(result_subset["per_class"].keys()) == {1, 3}
    assert result_all["per_class"][1] == pytest.approx(result_subset["per_class"][1])


def test_flattened_matrix_correlation_identical_matrices_gives_one() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(4, 8))
    assert flattened_matrix_correlation(matrix, matrix.copy()) == pytest.approx(1.0, abs=1e-8)


def test_flattened_matrix_correlation_ignores_nan_entries() -> None:
    matrix_a = np.array([[1.0, 2.0], [np.nan, np.nan]])
    matrix_b = np.array([[2.0, 4.0], [10.0, 20.0]])
    corr = flattened_matrix_correlation(matrix_a, matrix_b)
    assert corr == pytest.approx(1.0, abs=1e-8)
