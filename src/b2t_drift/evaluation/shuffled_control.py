"""Shuffled-input control: does the decoder use real neural information?

Replaces each trial's `input_features` with a randomly chosen, LENGTH-
COMPATIBLE donor trial's features from the SAME session, keeping the
original targets and session index. Tests whether the model is extracting
genuine neural signal from a session, or producing output about as
informative as feeding it the wrong trial's brain data -- raw PER alone
cannot distinguish "degraded decoding" from "the decoder has stopped using
neural information here" (`docs/DESIGN.md` section 4's attribution
question; `scripts/03_evaluate.py --shuffled-control`).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from b2t_drift.data.collate import collate_trials
from b2t_drift.data.hdf5_io import TrialId, read_trial


def build_shuffled_batches(
    trial_ids: list[TrialId],
    data_root: Path,
    session_to_index: dict[str, int],
    batch_size: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Build batches with each trial's features swapped for a donor's.

    Donor selection, for trial `T`: a candidate donor `D` (`D != T`, same
    session -- `trial_ids` is expected to already be scoped to one
    session) is LENGTH-COMPATIBLE if `D.n_time_steps >= T.seq_len` (CTC's
    hard input-length >= target-length requirement,
    `data/collate.py`'s module docstring). The donor's features are used
    in FULL, no truncation -- `T`'s own `seq_len` becomes the target
    length paired against the donor's full (possibly longer or shorter,
    as long as compatible) feature sequence. One donor drawn per trial via
    `random.Random(seed)`, for reproducibility.

    Args:
        trial_ids: Trials to build the shuffled control for, all from the
            SAME session (the donor candidate pool is exactly this list,
            not filtered internally).
        data_root: Directory containing per-session subdirectories.
        session_to_index: Session id -> day-layer index. Unchanged by
            shuffling: swapping features does not change which session's
            day-layer slice applies -- day-0 identity or trained, same as
            the real-features evaluation.
        batch_size: Trials per batch.
        seed: RNG seed for donor selection.

    Returns:
        `(batches, n_rejected)`: `batches` is a list of `collate_trials`
        output dicts covering every trial that found a length-compatible
        donor, in `trial_ids` order (rejected trials are simply absent,
        not replaced) -- so `batches` covers `len(trial_ids) - n_rejected`
        trials. `n_rejected` is the count of trials in `trial_ids` with NO
        length-compatible donor among the rest of `trial_ids`.
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    n_rejected = 0

    for t in trial_ids:
        candidates = [
            d for d in trial_ids
            if d.trial_key != t.trial_key and d.n_time_steps >= t.seq_len
        ]
        if not candidates:
            n_rejected += 1
            continue
        donor = rng.choice(candidates)
        donor_features, _ = read_trial(data_root, donor, validate=False)
        _, orig_targets = read_trial(data_root, t, validate=False)

        items.append({
            "features": torch.from_numpy(donor_features).float(),
            "targets": torch.from_numpy(orig_targets).long(),
            "n_time_steps": donor.n_time_steps,
            "seq_len": t.seq_len,
            "session_index": session_to_index[t.session_id],
            "session_id": t.session_id,
            "block_num": t.block_num,
            "trial_key": t.trial_key,
        })

    batches = [collate_trials(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]
    return batches, n_rejected
