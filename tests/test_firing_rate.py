"""Unit tests for `b2t_drift.analysis.firing_rate`."""

from __future__ import annotations

import numpy as np
import pytest

from b2t_drift.analysis.firing_rate import (
    BIN_WIDTH_SECONDS,
    recover_lattice_params,
    recover_raw_counts,
    trial_channel_firing_rate_hz,
)


def _synthetic_released_values(
    raw_counts: np.ndarray, sigma: float, mu: float
) -> np.ndarray:
    """Invert the module's own formula to build a fake released column:
    `released = (raw - mu) / sigma`, i.e. the released-value side of
    `raw = released * sigma + mu`.
    """
    return (raw_counts - mu) / sigma


def test_recovers_exact_sigma_and_mu_on_clean_synthetic_lattice() -> None:
    rng = np.random.default_rng(0)
    raw_counts = rng.integers(0, 50, size=500).astype(np.float64)
    true_sigma = 1.7823
    true_mu = -(raw_counts.min()) * true_sigma  # matches the module's own mu convention
    released = _synthetic_released_values(raw_counts, true_sigma, true_mu)

    result = recover_lattice_params(released)
    assert result.success
    assert result.sigma == pytest.approx(true_sigma, rel=1e-6)
    assert result.mu == pytest.approx(true_mu, rel=1e-6)


def test_recovers_lattice_with_missing_integer_counts() -> None:
    """Gaps in the integer sequence (e.g. count 2 never observed) must not
    break recovery, as long as every gap is still an integer multiple of
    the true spacing."""
    raw_counts = np.array([0, 1, 3, 3, 3, 5, 8, 8, 10, 15], dtype=np.float64)
    true_sigma = 0.6
    true_mu = -raw_counts.min() * true_sigma
    released = _synthetic_released_values(raw_counts, true_sigma, true_mu)

    result = recover_lattice_params(released)
    assert result.success
    assert result.sigma == pytest.approx(true_sigma, rel=1e-6)


def test_recover_raw_counts_matches_original_counts() -> None:
    rng = np.random.default_rng(1)
    raw_counts = rng.integers(0, 30, size=200).astype(np.float64)
    sigma, mu = 2.5, -3.0 * 2.5
    released = _synthetic_released_values(raw_counts, sigma, mu)

    lattice = recover_lattice_params(released)
    assert lattice.success
    recovered = recover_raw_counts(released, lattice)
    assert recovered == pytest.approx(raw_counts, abs=1e-6)


def test_fails_on_continuous_non_lattice_data() -> None:
    """Continuous data (e.g. SBP-like) must not spuriously report a
    lattice -- this is the SBP control check from docs/FINDINGS.md."""
    rng = np.random.default_rng(2)
    continuous = rng.normal(size=500)
    result = recover_lattice_params(continuous)
    assert not result.success
    assert result.reason is not None


def test_fails_with_too_few_distinct_values() -> None:
    values = np.array([1.0, 1.0, 1.0, 1.0])
    result = recover_lattice_params(values)
    assert not result.success
    assert "distinct value" in result.reason


def test_fails_when_lattice_is_noisy_beyond_tolerance() -> None:
    rng = np.random.default_rng(3)
    raw_counts = rng.integers(0, 50, size=300).astype(np.float64)
    sigma, mu = 1.0, 0.0
    released = _synthetic_released_values(raw_counts, sigma, mu)
    # Add enough per-sample jitter to break the lattice well beyond the
    # default relative tolerance.
    released = released + rng.normal(scale=0.3, size=released.shape)

    result = recover_lattice_params(released, relative_tolerance=1e-3)
    assert not result.success


def test_recover_raw_counts_raises_on_failed_lattice() -> None:
    rng = np.random.default_rng(4)
    continuous = rng.normal(size=100)
    lattice = recover_lattice_params(continuous)
    assert not lattice.success
    with pytest.raises(ValueError, match="did not succeed"):
        recover_raw_counts(continuous, lattice)


def test_trial_channel_firing_rate_hz_matches_expected_rate() -> None:
    # 20 Hz firing rate at 20 ms bins -> mean count/bin = 20 * 0.02 = 0.4
    rng = np.random.default_rng(5)
    raw_counts = rng.poisson(lam=0.4, size=2000).astype(np.float64)
    sigma, mu = 3.3, -raw_counts.min() * 3.3
    released = _synthetic_released_values(raw_counts, sigma, mu)

    rate_hz, lattice = trial_channel_firing_rate_hz(released)
    assert lattice.success
    assert rate_hz == pytest.approx(float(raw_counts.mean()) / BIN_WIDTH_SECONDS, rel=1e-6)
    assert rate_hz == pytest.approx(20.0, abs=3.0)  # loose bound on the Poisson draw itself


def test_trial_channel_firing_rate_hz_returns_none_on_failure() -> None:
    rng = np.random.default_rng(6)
    continuous = rng.normal(size=500)
    rate_hz, lattice = trial_channel_firing_rate_hz(continuous)
    assert rate_hz is None
    assert not lattice.success
