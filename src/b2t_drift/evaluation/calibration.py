"""Calibration/evaluation trial partitioning for Phase 3's recalibration arms.

PROTOCOL (docs/DESIGN.md section 5's "arms" -- Phase 3). For a given
evaluation session and a given number of calibration trials `N`, every arm
(Arm 0 through Arm 3b) must be compared on the SAME partition of that
session's trials into `N` calibration trials and the rest as evaluation
trials -- never let one arm see a different split than another for the same
(session, N, split index). `calibration_split_seed` is the single place that
maps `(session_id, n_calibration, split_index)` to a deterministic seed, so
every caller that needs "the same partition" for a given cell gets it by
construction, without passing the actual trial lists around.

THE UNIT OF LEAKAGE IS THE SENTENCE, NOT THE TRIAL. The rule "never evaluate
on a calibration TRIAL" is not enough: the same sentence text can appear as
two distinct trials in one session, so a model fine-tuned on one can be scored
on the other and get credit for memorizing a phoneme sequence rather than
decoding it. The correct rule is "never evaluate on a SENTENCE used for
calibration." `split_calibration_trials(..., dedup_sentences=True)` enforces
it. The DEFAULT is still `False` (trial-level exclusion only) -- deliberately,
so existing results stay reproducible; the Phase 3 recalibration grid was
built with `dedup_sentences=False`, and its sentence-overlap-corrected values
are recorded alongside the originals in `results/recalibration_grid.csv`.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Sequence

from b2t_drift.data.hdf5_io import TrialId

DEFAULT_BIN_MS = 20.0


def normalize_sentence(label: str) -> str:
    """Canonical form of a `sentence_label` for equality comparison:
    case-folded with runs of whitespace collapsed. Punctuation is left
    alone -- two labels differing only by punctuation are treated as
    different sentences (conservative in the direction of NOT discarding
    eval trials; `docs/DATASET.md` labels are consistently punctuated)."""
    return " ".join(label.lower().split())


def split_off_sentence_overlap(
    calibration: Sequence[TrialId],
    evaluation: Sequence[TrialId],
) -> tuple[list[TrialId], list[TrialId]]:
    """Split `evaluation` into `(kept, removed)`, where `removed` are trials
    whose sentence text (`normalize_sentence`) also appears anywhere in
    `calibration`.

    Both lists keep `evaluation`'s relative order. `calibration` is never
    modified. Either output may be empty.
    """
    calibration_sentences = {normalize_sentence(t.sentence_label) for t in calibration}
    kept: list[TrialId] = []
    removed: list[TrialId] = []
    for t in evaluation:
        (removed if normalize_sentence(t.sentence_label) in calibration_sentences else kept).append(t)
    return kept, removed


def calibration_split_seed(
    session_id: str,
    n_calibration: int,
    split_index: int,
    base_seed: int = 0,
) -> int:
    """Deterministic seed for one (session, N, split) calibration partition.

    CRC32 over a plain string encoding of the four inputs -- not
    cryptographic, just a cheap, stable way to turn an arbitrary
    `session_id` string plus two small integers into a well-spread 32-bit
    seed, so cells that differ in ANY of `n_calibration`, `split_index`, or
    `session_id` get independent-looking partitions, while the SAME triple
    always reproduces the SAME seed (and therefore the same partition, via
    `split_calibration_trials`) -- this is what makes the whole grid
    reproducible from `base_seed` alone.

    Args:
        session_id: The evaluation session this partition is for.
        n_calibration: N for this cell.
        split_index: Which of the (typically 5) repeated splits this is.
        base_seed: Project-wide base seed, so the entire grid can be
            re-derived from one number recorded in results/*.csv.

    Returns:
        A non-negative int suitable for `random.Random(seed)`.
    """
    key = f"{base_seed}|{session_id}|{n_calibration}|{split_index}"
    return zlib.crc32(key.encode("utf-8"))


def split_calibration_trials(
    trial_ids: Sequence[TrialId],
    n_calibration: int,
    seed: int,
    dedup_sentences: bool = False,
) -> tuple[list[TrialId], list[TrialId]]:
    """Partition one session's trials into `n_calibration` calibration
    trials and the rest as evaluation trials.

    Args:
        trial_ids: All of one session's labeled trials, in any order.
        n_calibration: Number of trials to hold out for calibration. Must
            leave at least one trial for evaluation (`n_calibration <
            len(trial_ids)`) -- a cell that cannot satisfy this (e.g. N=100
            on a ~94-trial session) is IMPOSSIBLE and must be skipped by the
            caller before this function is ever called, not silently
            reduced here (docs/DESIGN.md's Phase 3 instructions: "SKIP such
            cells... Do not silently reduce N or reuse calibration trials").
        seed: From `calibration_split_seed`, or any int -- callers that need
            the standard, reproducible grid should always go through that
            function rather than inventing their own seed.
        dedup_sentences: If True, additionally DROP from the evaluation set
            every trial whose sentence text also appears in the calibration
            set (`split_off_sentence_overlap`). The calibration set is the
            same either way for a given seed, so a dedup'd evaluation set is
            always a subset of the non-dedup'd one -- and in that case the
            two returned lists no longer partition `trial_ids` (the dropped
            trials belong to neither). Default False keeps the original
            trial-level-only behavior (what the Phase 3 grid used).

    Returns:
        `(calibration_trials, evaluation_trials)`, both lists of `TrialId`,
        each a subsequence of `trial_ids` in ITS original relative order
        (not shuffled) -- only the assignment to the two groups is random,
        chosen by `random.Random(seed).sample`. The two lists partition
        `trial_ids` exactly: every element of `trial_ids` appears in
        exactly one of them, and neither list is ever empty when
        `0 < n_calibration < len(trial_ids)` (with `dedup_sentences=True` the
        evaluation list can be shorter, or even empty if every remaining
        sentence was also calibrated on -- callers must check).

    Raises:
        ValueError: If `n_calibration < 0`, or `n_calibration >=
            len(trial_ids)` (no trials would be left to evaluate on).
    """
    n_total = len(trial_ids)
    if n_calibration < 0:
        raise ValueError(
            f"split_calibration_trials: n_calibration={n_calibration} must be >= 0."
        )
    if n_calibration >= n_total:
        raise ValueError(
            f"split_calibration_trials: n_calibration={n_calibration} >= "
            f"len(trial_ids)={n_total} -- no trials would remain for "
            f"evaluation. This cell is IMPOSSIBLE and must be skipped by "
            f"the caller, not reduced here."
        )

    rng = random.Random(seed)
    calibration_indices = set(rng.sample(range(n_total), n_calibration))

    calibration = [t for i, t in enumerate(trial_ids) if i in calibration_indices]
    evaluation = [t for i, t in enumerate(trial_ids) if i not in calibration_indices]
    if dedup_sentences:
        evaluation, _ = split_off_sentence_overlap(calibration, evaluation)
    return calibration, evaluation


def calibration_minutes(trial_ids: Sequence[TrialId], bin_ms: float = DEFAULT_BIN_MS) -> float:
    """Total participant time represented by a set of trials, in minutes.

    `n_time_steps * bin_ms` is each trial's real (unpadded) duration in
    milliseconds (docs/DATASET.md's 20 ms bins) -- this is the axis Phase 3
    actually cares about, since "N calibration sentences" alone doesn't say
    how much of the participant's time that cost (a 3-word sentence and a
    20-word sentence are very different asks).

    Args:
        trial_ids: Trials to sum over (typically one cell's calibration set).
        bin_ms: Milliseconds per timestep. 20.0, matching every other use
            of this constant in the project (docs/DATASET.md section 9).

    Returns:
        Total duration in minutes, `0.0` for an empty sequence.
    """
    total_ms = sum(t.n_time_steps for t in trial_ids) * bin_ms
    return total_ms / 1000.0 / 60.0
