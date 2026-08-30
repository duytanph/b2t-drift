"""Aggregate evaluation metrics.

Responsible for combining per-example `EditCounts` (from
`evaluation.edit_distance`) into corpus-level metrics such as overall
phoneme error rate, and for organizing metrics by session for the
drift analysis.
"""

from __future__ import annotations

from b2t_drift.evaluation.edit_distance import EditCounts


def aggregate_error_rate(counts: list[EditCounts]) -> float:
    """Compute a corpus-level error rate from per-example edit counts.

    Args:
        counts: One `EditCounts` per evaluated example.

    Returns:
        Total edits across all examples, divided by total reference length
        (not the mean of per-example error rates).
    """
    raise NotImplementedError


def error_rate_by_session(
    counts_by_session: dict[str, list[EditCounts]],
) -> dict[str, float]:
    """Compute a corpus-level error rate per session.

    Args:
        counts_by_session: Session id -> list of per-example `EditCounts`.

    Returns:
        Session id -> aggregate error rate for that session.
    """
    raise NotImplementedError
