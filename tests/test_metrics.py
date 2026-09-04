"""Unit tests for `b2t_drift.evaluation.metrics`.

Targets Task 4: `aggregate_error_rate` (implemented as documented),
`bootstrap_error_rate_ci` (example-level resampling, reproducibility, CI
width shrinking with more examples), and `per_session_metrics` (sorted
output, derived fields, required-metadata errors).
"""

from __future__ import annotations

import random

import pytest

from b2t_drift.evaluation.edit_distance import EditCounts
from b2t_drift.evaluation.metrics import (
    aggregate_error_rate,
    bootstrap_error_rate_ci,
    error_rate_by_session,
    per_session_metrics,
)


def _synthetic_counts(n_examples: int, seed: int, error_rate: float = 0.15) -> list[EditCounts]:
    """Examples with varying reference lengths and a roughly fixed
    per-phoneme error probability, for tests that need genuine sampling
    variance in the aggregate rate."""
    rng = random.Random(seed)
    counts = []
    for _ in range(n_examples):
        ref_len = rng.randint(10, 30)
        errors = sum(1 for _ in range(ref_len) if rng.random() < error_rate)
        counts.append(EditCounts(substitutions=errors, deletions=0, insertions=0, reference_length=ref_len))
    return counts


# --- aggregate_error_rate ------------------------------------------------


def test_aggregate_rate_differs_from_mean_of_per_example_rates() -> None:
    # One short, high-error example and one long, low-error example.
    counts = [
        EditCounts(substitutions=4, deletions=0, insertions=0, reference_length=5),   # rate 0.8
        EditCounts(substitutions=2, deletions=0, insertions=0, reference_length=50),  # rate 0.04
    ]
    aggregate = aggregate_error_rate(counts)
    mean_of_rates = sum(c.error_rate for c in counts) / len(counts)

    assert aggregate == pytest.approx(6 / 55)
    assert mean_of_rates == pytest.approx((0.8 + 0.04) / 2)
    assert aggregate != pytest.approx(mean_of_rates)


def test_aggregate_rate_above_100_percent_is_handled() -> None:
    counts = [EditCounts(substitutions=0, deletions=0, insertions=8, reference_length=5)]
    rate = aggregate_error_rate(counts)
    assert rate == pytest.approx(8 / 5)
    assert rate > 1.0


def test_aggregate_rate_empty_counts_is_zero() -> None:
    assert aggregate_error_rate([]) == 0.0


# --- bootstrap_error_rate_ci ----------------------------------------------


def test_bootstrap_ci_brackets_point_estimate() -> None:
    counts = _synthetic_counts(100, seed=1)
    point, lower, upper = bootstrap_error_rate_ci(counts, n_resamples=2000, seed=0)
    assert lower <= point <= upper


def test_bootstrap_is_reproducible_under_fixed_seed() -> None:
    counts = _synthetic_counts(50, seed=2)
    result_a = bootstrap_error_rate_ci(counts, n_resamples=1000, seed=42)
    result_b = bootstrap_error_rate_ci(counts, n_resamples=1000, seed=42)
    assert result_a == result_b


def test_bootstrap_different_seeds_can_differ() -> None:
    counts = _synthetic_counts(50, seed=2)
    result_a = bootstrap_error_rate_ci(counts, n_resamples=1000, seed=1)
    result_b = bootstrap_error_rate_ci(counts, n_resamples=1000, seed=2)
    assert result_a != result_b


def test_bootstrap_ci_width_shrinks_as_example_count_grows() -> None:
    small = _synthetic_counts(80, seed=7)
    large = _synthetic_counts(360, seed=7)

    _, lo_small, hi_small = bootstrap_error_rate_ci(small, n_resamples=3000, seed=0)
    _, lo_large, hi_large = bootstrap_error_rate_ci(large, n_resamples=3000, seed=0)

    width_small = hi_small - lo_small
    width_large = hi_large - lo_large
    assert width_large < width_small


def test_bootstrap_raises_on_empty_counts() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_error_rate_ci([])


def test_bootstrap_raises_on_invalid_confidence() -> None:
    counts = _synthetic_counts(10, seed=1)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_error_rate_ci(counts, confidence=1.5)


# --- error_rate_by_session --------------------------------------------------


def test_error_rate_by_session_matches_aggregate_per_session() -> None:
    counts_by_session = {
        "s1": _synthetic_counts(20, seed=1),
        "s2": _synthetic_counts(20, seed=2),
    }
    result = error_rate_by_session(counts_by_session)
    assert result["s1"] == aggregate_error_rate(counts_by_session["s1"])
    assert result["s2"] == aggregate_error_rate(counts_by_session["s2"])


# --- per_session_metrics ----------------------------------------------------


def test_per_session_metrics_sorted_by_days_since_cut() -> None:
    counts_by_session = {
        "late": _synthetic_counts(15, seed=1),
        "early": _synthetic_counts(15, seed=2),
        "mid": _synthetic_counts(15, seed=3),
    }
    session_metadata = {
        "late": {"session_date": "2024-01-01", "days_since_cut": 300},
        "early": {"session_date": "2023-09-01", "days_since_cut": 5},
        "mid": {"session_date": "2023-11-01", "days_since_cut": 60},
    }
    rows = per_session_metrics(counts_by_session, session_metadata, n_resamples=200)
    assert [r["session_id"] for r in rows] == ["early", "mid", "late"]
    assert [r["days_since_cut"] for r in rows] == [5, 60, 300]


def test_per_session_metrics_derived_fields() -> None:
    counts = [
        EditCounts(substitutions=1, deletions=0, insertions=0, reference_length=10),
        EditCounts(substitutions=0, deletions=1, insertions=0, reference_length=20),
        EditCounts(substitutions=2, deletions=0, insertions=1, reference_length=30),
    ]
    counts_by_session = {"s1": counts}
    session_metadata = {"s1": {"session_date": "2023-09-01", "days_since_cut": 5}}

    rows = per_session_metrics(counts_by_session, session_metadata, n_resamples=200)
    row = rows[0]

    assert row["n_trials"] == 3
    assert row["median_seq_len"] == 20.0  # median of [10, 20, 30]
    assert row["n_reference_phonemes"] == 60  # 10+20+30
    assert row["per"] == aggregate_error_rate(counts)
    assert row["per_ci_lower"] <= row["per"] <= row["per_ci_upper"]


def test_per_session_metrics_raises_on_missing_metadata() -> None:
    counts_by_session = {"s1": _synthetic_counts(5, seed=1)}
    with pytest.raises(ValueError, match="s1"):
        per_session_metrics(counts_by_session, session_metadata={})


def test_per_session_metrics_raises_on_empty_session_counts() -> None:
    counts_by_session = {"s1": []}
    session_metadata = {"s1": {"session_date": "2023-09-01", "days_since_cut": 5}}
    with pytest.raises(ValueError, match="s1"):
        per_session_metrics(counts_by_session, session_metadata)
