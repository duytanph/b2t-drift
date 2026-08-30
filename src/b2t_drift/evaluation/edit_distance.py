"""Levenshtein alignment between reference and hypothesis token sequences.

This is the core error-counting primitive used throughout evaluation: given
a reference and hypothesis sequence of tokens (e.g. phoneme symbols or
label indices), compute the minimum-edit-distance alignment and return the
counts of substitutions, deletions, and insertions, plus the reference
length (for normalizing into an error rate).

This module is fully implemented and self-contained -- it has no dependency
on the rest of b2t_drift.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


class EditCounts(NamedTuple):
    """Counts from a Levenshtein alignment between reference and hypothesis."""

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def error_rate(self) -> float:
        """Edit distance normalized by reference length.

        Note: this can exceed 1.0 (100%) when the hypothesis has enough
        insertions relative to a short reference. When the reference is
        empty, the error rate is 0.0 if the hypothesis is also empty
        (no errors), and infinite otherwise.
        """
        total_errors = self.substitutions + self.deletions + self.insertions
        if self.reference_length == 0:
            return 0.0 if total_errors == 0 else float("inf")
        return total_errors / self.reference_length


def levenshtein_alignment(
    reference: Sequence[object],
    hypothesis: Sequence[object],
) -> EditCounts:
    """Compute the minimum-edit-distance alignment between two sequences.

    Uses the standard dynamic-programming Levenshtein formulation with unit
    costs for substitution, deletion, and insertion. When multiple
    minimum-cost alignments exist, ties are broken in the order: match,
    substitution, deletion, insertion (i.e. the backtrace prefers a match
    if available, then a substitution, then a deletion, before falling
    back to an insertion).

    Args:
        reference: The reference (ground-truth) token sequence.
        hypothesis: The hypothesis (predicted) token sequence.

    Returns:
        An `EditCounts` with the number of substitutions, deletions, and
        insertions in the chosen alignment, plus `len(reference)`.
    """
    n = len(reference)
    m = len(hypothesis)

    # dp[i][j] = min edit distance between reference[:i] and hypothesis[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],  # deletion (reference token dropped)
                    dp[i][j - 1],  # insertion (hypothesis token added)
                )

    # Backtrace from (n, m) to (0, 0), counting operation types. Ties are
    # broken in the order: match, substitution, deletion, insertion.
    substitutions = 0
    deletions = 0
    insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
            continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue
        raise AssertionError("unreachable: inconsistent DP table during backtrace")

    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=n,
    )
