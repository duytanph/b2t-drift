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
    patch_size: int = 1,
    patch_stride: int = 1,
    feature_subset: slice | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Build batches with each trial's features swapped for a donor's.

    Donor selection, for trial `T`: a candidate donor `D` (`D != T`, same
    session -- `trial_ids` is expected to already be scoped to one
    session) is LENGTH-COMPATIBLE if the donor's feature sequence, AFTER
    the model's temporal downsampling, is still at least `T.seq_len` long
    (CTC's hard input-length >= target-length requirement,
    `data/collate.py`'s module docstring). The donor's features are used
    in FULL, no truncation -- `T`'s own `seq_len` becomes the target
    length paired against the donor's full feature sequence. One donor
    drawn per trial via `random.Random(seed)`, for reproducibility.

    THE PATCHING CHECK. `patch_size`/`patch_stride` mirror
    `GRUCTC.__init__`'s (default `1`/`1` = no downsampling, and the check
    reduces to the raw `D.n_time_steps >= T.seq_len` it was before
    patching existed). When the model patches its input
    (`patch_size=14, patch_stride=4`), a donor whose raw `n_time_steps`
    only just clears `T.seq_len` collapses to `~n_time_steps/4` patches --
    below `T.seq_len` -- and CTC then cannot emit enough tokens, forcing a
    near-empty decode and a spuriously high shuffled PER. This is
    `CLAUDE.md`'s "known pitfall" (`collate_trials`' `input_lengths`
    predate model-side downsampling) applied to donor selection: the
    candidate filter uses the post-downsample length, via the SAME formula
    as `GRUCTC.ctc_input_lengths`.

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
        patch_size, patch_stride: The evaluated model's patching
            parameters (`GRUCTC.patch_size` / `.patch_stride`). Pass them
            whenever the model downsamples the time axis, or a donor long
            enough raw but too short once patched slips through and
            inflates that session's shuffled PER.
        feature_subset: If given (e.g. `data.dataset.FEATURE_SUBSETS["tx"]`),
            slice both the donor's and (implicitly, via the same column
            range) the evaluated model's expected feature width down to
            this subset -- must match whatever `feature_subset` the
            evaluated model's `TrialDataset` was built with, or the
            shuffled control silently feeds the model a different feature
            set than its real-data evaluation used. `None` (the default):
            all 512 columns, unchanged behavior.

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

    def _downsampled_len(n_time_steps: int) -> int:
        # Identical to GRUCTC.ctc_input_lengths (identity when 1/1).
        if patch_size == 1 and patch_stride == 1:
            return n_time_steps
        return (n_time_steps - patch_size) // patch_stride + 1

    for t in trial_ids:
        candidates = [
            d for d in trial_ids
            if d.trial_key != t.trial_key and _downsampled_len(d.n_time_steps) >= t.seq_len
        ]
        if not candidates:
            n_rejected += 1
            continue
        donor = rng.choice(candidates)
        donor_features, _ = read_trial(data_root, donor, validate=False)
        _, orig_targets = read_trial(data_root, t, validate=False)

        donor_features_tensor = torch.from_numpy(donor_features).float()
        if feature_subset is not None:
            donor_features_tensor = donor_features_tensor[:, feature_subset]

        items.append({
            "features": donor_features_tensor,
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
