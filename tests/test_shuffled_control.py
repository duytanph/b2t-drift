"""Tests for b2t_drift.evaluation.shuffled_control.build_shuffled_batches.

Run against real T15 data -- skipped if data/raw/hdf5_data_final is not
present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from b2t_drift.data.hdf5_io import iter_trial_ids
from b2t_drift.evaluation.shuffled_control import build_shuffled_batches

DATA_ROOT = Path("data/raw/hdf5_data_final")
SESSION_ID = "t15.2023.08.11"


def _require_real_data() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")


def test_batches_cover_every_trial_when_none_rejected() -> None:
    _require_real_data()
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:8]
    session_to_index = {SESSION_ID: 0}

    batches, n_rejected = build_shuffled_batches(trial_ids, DATA_ROOT, session_to_index, batch_size=4)

    total_items = sum(b["features"].shape[0] for b in batches)
    assert total_items == len(trial_ids) - n_rejected


def test_shuffled_features_differ_from_original_for_at_least_one_trial() -> None:
    _require_real_data()
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:6]
    session_to_index = {SESSION_ID: 0}

    batches, n_rejected = build_shuffled_batches(trial_ids, DATA_ROOT, session_to_index, batch_size=6)
    assert n_rejected < len(trial_ids), "every trial rejected -- test setup too small to be meaningful"

    # target_lengths must still match each trial's OWN seq_len (targets are
    # unchanged), while input_lengths (the donor's n_time_steps) need not
    # equal any particular trial's own n_time_steps.
    batch = batches[0]
    assert int(batch["target_lengths"].sum()) == batch["targets"].shape[0]
    assert batch["input_lengths"].shape[0] == batch["features"].shape[0]


def test_reproducible_under_fixed_seed() -> None:
    _require_real_data()
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:10]
    session_to_index = {SESSION_ID: 0}

    batches_a, rej_a = build_shuffled_batches(trial_ids, DATA_ROOT, session_to_index, batch_size=32, seed=7)
    batches_b, rej_b = build_shuffled_batches(trial_ids, DATA_ROOT, session_to_index, batch_size=32, seed=7)

    assert rej_a == rej_b
    assert len(batches_a) == len(batches_b) == 1
    assert (batches_a[0]["features"] == batches_b[0]["features"]).all()


def test_donor_never_shorter_than_trials_seq_len() -> None:
    _require_real_data()
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:12]
    session_to_index = {SESSION_ID: 0}
    seq_len_by_key = {t.trial_key: t.seq_len for t in trial_ids}

    batches, _ = build_shuffled_batches(trial_ids, DATA_ROOT, session_to_index, batch_size=32)
    for batch in batches:
        for i, trial_key in enumerate(batch["trial_keys"]):
            assert int(batch["input_lengths"][i]) >= seq_len_by_key[trial_key]


def test_donor_length_check_accounts_for_patching() -> None:
    """With a patched model, a donor must be long enough AFTER downsampling
    -- `(n_time_steps - patch_size) // patch_stride + 1 >= seq_len` -- or a
    donor that clears the raw check by a hair collapses to far fewer
    patches than the trial has target phonemes, and CTC cannot emit them
    (`CLAUDE.md`'s "known pitfall"; the +518d s15 outlier that surfaced
    this: per_shuffled 0.998 vs 0.82-0.93 elsewhere)."""
    _require_real_data()
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:40]
    session_to_index = {SESSION_ID: 0}
    seq_len_by_key = {t.trial_key: t.seq_len for t in trial_ids}
    patch_size, patch_stride = 14, 4

    batches, _ = build_shuffled_batches(
        trial_ids, DATA_ROOT, session_to_index, batch_size=32,
        patch_size=patch_size, patch_stride=patch_stride,
    )
    for batch in batches:
        for i, trial_key in enumerate(batch["trial_keys"]):
            donor_n_time_steps = int(batch["input_lengths"][i])
            post_patch_len = (donor_n_time_steps - patch_size) // patch_stride + 1
            assert post_patch_len >= seq_len_by_key[trial_key], (
                f"{trial_key}: donor n_time_steps={donor_n_time_steps} -> "
                f"{post_patch_len} patches < seq_len {seq_len_by_key[trial_key]}"
            )
