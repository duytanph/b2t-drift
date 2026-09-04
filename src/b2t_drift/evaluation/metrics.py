"""Aggregate evaluation metrics.

Responsible for combining per-example `EditCounts` (from
`evaluation.edit_distance`) into corpus-level metrics such as overall
phoneme error rate, and for organizing metrics by session for the
drift analysis.

WHY EXAMPLE-LEVEL BOOTSTRAP RESAMPLING, NOT PHONEME-LEVEL. Held-out
sessions range from about 80 to 364 trials (`docs/DATASET.md` section 7 /
13) -- PER on 80 sentences is noisy enough that random variation would
read as structure on a drift curve unless every point carries an
interval. `bootstrap_error_rate_ci` resamples EXAMPLES (trials) with
replacement, not individual phonemes: the unit of independence in this
data is the trial (each trial's CTC decode and its errors are one
correlated blob -- a substitution early in a sentence is not independent
of the sentence's other phonemes), so resampling at the phoneme level
would treat correlated errors within a trial as independent draws and
understate the true uncertainty. Resampling whole trials, and
recomputing the AGGREGATE (not mean-of-rates) error rate each time,
preserves that correlation structure in every bootstrap replicate.

WHY AGGREGATE, NOT THE MEAN OF PER-EXAMPLE RATES. `aggregate_error_rate`
is total edits over total reference length across every example, not the
arithmetic mean of each example's own edit distance divided by its own
reference length. These differ whenever examples have unequal reference
lengths (nearly always true here -- `seq_len` ranges 5-47,
`docs/DATASET.md` section 6): the mean of per-example rates weights a
5-phoneme example and a 47-phoneme example equally, while the aggregate
rate weights every PHONEME equally regardless of which example it came
from. The aggregate is the standard, and correct, way to report a
corpus-level PER for exactly this reason -- see
`tests/test_metrics.py::test_aggregate_rate_differs_from_mean_of_per_example_rates`
for a concrete case where the two numbers disagree.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

import numpy as np

from b2t_drift.evaluation.edit_distance import EditCounts


def aggregate_error_rate(counts: list[EditCounts]) -> float:
    """Compute a corpus-level error rate from per-example edit counts.

    Args:
        counts: One `EditCounts` per evaluated example.

    Returns:
        Total edits across all examples, divided by total reference length
        (not the mean of per-example error rates -- see the module
        docstring for why that distinction matters). If total reference
        length is 0 (e.g. `counts` is empty, or every example had an
        empty reference), returns 0.0 if there were also no edits, or
        `float("inf")` otherwise -- matching `EditCounts.error_rate`'s
        own convention for the same degenerate case. A rate above 1.0
        (100%) is returned as-is, not clamped: it is a legitimate outcome
        when a hypothesis has enough insertions relative to a short
        reference (see `EditCounts.error_rate`'s docstring), and
        silently clamping it would hide that from a caller.
    """
    total_edits = sum(c.substitutions + c.deletions + c.insertions for c in counts)
    total_reference_length = sum(c.reference_length for c in counts)
    if total_reference_length == 0:
        return 0.0 if total_edits == 0 else float("inf")
    return total_edits / total_reference_length


def bootstrap_error_rate_ci(
    counts: list[EditCounts],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval around the aggregate error rate.

    Resamples EXAMPLES (not individual phonemes) with replacement -- see
    the module docstring for why that unit of resampling matters here --
    recomputing `aggregate_error_rate` on each resample to build an
    empirical distribution of the aggregate rate, then reads off the
    percentile interval from that distribution. MANDATORY for the drift
    curve (`docs/DESIGN.md` section 7): every point needs an interval,
    since the smallest held-out sessions have as few as ~80 trials.

    Args:
        counts: One `EditCounts` per evaluated example (trial). Must be
            non-empty.
        n_resamples: Number of bootstrap resamples to draw. 10,000 is
            generous for a 95% interval on up to a few hundred examples;
            reduce for faster, less precise intervals (e.g. in tests).
        confidence: Interval width, e.g. 0.95 for a 95% CI (2.5th to
            97.5th percentile of the resampled rates).
        seed: Seed for the resampling RNG (`random.Random(seed)`, a
            local instance -- does not touch or get affected by global
            `random` state, matching this project's convention elsewhere,
            e.g. `data/splits.py`). Fixing this makes the returned
            interval exactly reproducible for a given `counts` and
            `n_resamples`.

    Returns:
        `(point_estimate, ci_lower, ci_upper)`: `point_estimate` is
        `aggregate_error_rate(counts)` on the ORIGINAL (non-resampled)
        data; `ci_lower`/`ci_upper` bracket it at the requested
        `confidence` level, computed from the resampled distribution.
        `ci_lower <= point_estimate <= ci_upper` is not a hard
        mathematical guarantee of any bootstrap procedure (the point
        estimate can in principle fall outside a percentile interval
        built from resamples), but is true in practice for a reasonably
        sized, non-degenerate `counts` list -- see
        `tests/test_metrics.py::test_bootstrap_ci_brackets_point_estimate`.

    Raises:
        ValueError: If `counts` is empty (there is nothing to resample),
            or `confidence` is not strictly between 0 and 1.
    """
    if not counts:
        raise ValueError(
            "bootstrap_error_rate_ci: counts is empty -- there is nothing "
            "to resample."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"bootstrap_error_rate_ci: confidence={confidence} must be "
            f"strictly between 0 and 1."
        )

    point_estimate = aggregate_error_rate(counts)

    n = len(counts)
    rng = random.Random(seed)
    resampled_rates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        resample = [counts[rng.randrange(n)] for _ in range(n)]
        resampled_rates[i] = aggregate_error_rate(resample)

    alpha = 1.0 - confidence
    lower, upper = np.percentile(resampled_rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point_estimate, float(lower), float(upper)


def error_rate_by_session(
    counts_by_session: dict[str, list[EditCounts]],
) -> dict[str, float]:
    """Compute a corpus-level error rate per session.

    Args:
        counts_by_session: Session id -> list of per-example `EditCounts`.

    Returns:
        Session id -> `aggregate_error_rate` for that session's counts.
        No confidence interval here -- use `per_session_metrics` (or call
        `bootstrap_error_rate_ci` directly per session) when an interval
        is needed, e.g. for the drift curve.
    """
    return {
        session_id: aggregate_error_rate(counts)
        for session_id, counts in counts_by_session.items()
    }


def per_session_metrics(
    counts_by_session: dict[str, list[EditCounts]],
    session_metadata: dict[str, dict[str, Any]],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build one drift-curve row per session, sorted by `days_since_cut`.

    Args:
        counts_by_session: Session id -> list of per-TRIAL `EditCounts`
            for that session's evaluated trials (one `EditCounts` per
            trial, NOT pre-aggregated). This is what lets this function
            derive `n_trials`, the per-session PER and its bootstrap CI,
            AND `median_seq_len` / `n_reference_phonemes` all from one
            input: `EditCounts.reference_length` is exactly that trial's
            ground-truth phoneme count (`evaluation.edit_distance`), so
            nothing here needs to be passed in twice.
        session_metadata: Session id -> `{"session_date": "YYYY-MM-DD",
            "days_since_cut": int}`. Only these two fields are required
            -- everything else in the output is derived from
            `counts_by_session`, so this function does not accept, and
            would not use, a redundant `n_trials` or `median_seq_len` in
            `session_metadata` that could silently disagree with the
            counts (this project's split-manifest / corpus-CSV modules
            follow the same "one field, one source of truth" rule --
            `data/splits.py`'s module docstring).
        n_resamples, confidence, seed: Passed straight through to
            `bootstrap_error_rate_ci` for every session.

    Returns:
        One dict per session id present in `counts_by_session`, with keys
        `session_id`, `session_date`, `days_since_cut`, `n_trials`, `per`,
        `per_ci_lower`, `per_ci_upper`, `median_seq_len`,
        `n_reference_phonemes`. **Sorted by `days_since_cut` ascending**
        -- this is the drift curve's x-axis; an unsorted result would
        silently produce a scrambled plot rather than a curve.

    Raises:
        ValueError: If a session id in `counts_by_session` has an empty
            `EditCounts` list (PER is undefined for zero trials -- naming
            the session), or is missing from `session_metadata` (naming
            the session).
    """
    rows: list[dict[str, Any]] = []
    for session_id, counts in counts_by_session.items():
        if not counts:
            raise ValueError(
                f"per_session_metrics: session {session_id!r} has an "
                f"empty EditCounts list -- PER is undefined for zero "
                f"trials."
            )
        if session_id not in session_metadata:
            raise ValueError(
                f"per_session_metrics: session {session_id!r} is in "
                f"counts_by_session but has no entry in session_metadata."
            )
        meta = session_metadata[session_id]
        point, lower, upper = bootstrap_error_rate_ci(
            counts, n_resamples=n_resamples, confidence=confidence, seed=seed,
        )
        reference_lengths = [c.reference_length for c in counts]
        rows.append({
            "session_id": session_id,
            "session_date": meta["session_date"],
            "days_since_cut": meta["days_since_cut"],
            "n_trials": len(counts),
            "per": point,
            "per_ci_lower": lower,
            "per_ci_upper": upper,
            "median_seq_len": float(statistics.median(reference_lengths)),
            "n_reference_phonemes": sum(reference_lengths),
        })

    rows.sort(key=lambda r: r["days_since_cut"])
    return rows
