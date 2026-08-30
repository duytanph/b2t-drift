"""Tests for levenshtein_alignment in b2t_drift.evaluation.edit_distance."""

from __future__ import annotations

from b2t_drift.evaluation.edit_distance import EditCounts, levenshtein_alignment


def test_identical_sequences_have_no_errors() -> None:
    ref = ["s", "t", "aa", "p"]
    hyp = ["s", "t", "aa", "p"]

    result = levenshtein_alignment(ref, hyp)

    assert result == EditCounts(substitutions=0, deletions=0, insertions=0, reference_length=4)
    assert result.error_rate == 0.0


def test_single_substitution() -> None:
    ref = ["s", "t", "aa", "p"]
    hyp = ["s", "t", "iy", "p"]

    result = levenshtein_alignment(ref, hyp)

    assert result == EditCounts(substitutions=1, deletions=0, insertions=0, reference_length=4)


def test_empty_hypothesis_counts_all_deletions() -> None:
    ref = ["s", "t", "aa", "p"]
    hyp: list[str] = []

    result = levenshtein_alignment(ref, hyp)

    assert result.substitutions == 0
    assert result.insertions == 0
    assert result.deletions == 4
    assert result.reference_length == 4
    assert result.error_rate == 1.0


def test_empty_reference_and_hypothesis_has_zero_error_rate() -> None:
    result = levenshtein_alignment([], [])

    assert result == EditCounts(substitutions=0, deletions=0, insertions=0, reference_length=0)
    assert result.error_rate == 0.0


def test_hypothesis_longer_than_reference_counts_insertions() -> None:
    ref = ["s", "t"]
    hyp = ["s", "t", "aa", "p", "l"]

    result = levenshtein_alignment(ref, hyp)

    assert result.substitutions == 0
    assert result.deletions == 0
    assert result.insertions == 3
    assert result.reference_length == 2


def test_error_rate_can_exceed_one_hundred_percent() -> None:
    # A short reference with a completely different, longer hypothesis
    # forces both substitutions and insertions, pushing the error rate
    # above 1.0 (100%).
    ref = ["s", "t"]
    hyp = ["b", "d", "k", "g"]

    result = levenshtein_alignment(ref, hyp)

    assert result.reference_length == 2
    assert (result.substitutions + result.deletions + result.insertions) > result.reference_length
    assert result.error_rate > 1.0
