"""Tests for the array-reading half of b2t_drift.data.hdf5_io.

Run against the REAL T15 data files (not fixtures) -- the whole module is
skipped if data/raw/hdf5_data_final is not present, e.g. on a machine that
hasn't downloaded the dataset (see README.md "Setup" / "Data").
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from b2t_drift.data.hdf5_io import iter_trial_ids, read_trial, read_trials

DATA_ROOT = Path("data/raw/hdf5_data_final")
MANIFEST_PATH = Path("data/interim/session_manifest.csv")


@pytest.fixture(autouse=True)
def _require_real_data() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")


def test_iter_trial_ids_count_t15_2023_08_11() -> None:
    ids = list(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))
    assert len(ids) == 288


def test_read_trial_features_shape_and_dtype() -> None:
    trial_id = next(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))
    features, _ = read_trial(DATA_ROOT, trial_id)
    assert features.shape == (trial_id.n_time_steps, 512)
    assert features.dtype == np.float32


def test_read_trial_targets_shape() -> None:
    trial_id = next(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))
    _, targets = read_trial(DATA_ROOT, trial_id)
    assert targets.shape == (trial_id.seq_len,)


def test_all_targets_in_range_1_to_40() -> None:
    for trial_id in list(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))[:50]:
        _, targets = read_trial(DATA_ROOT, trial_id)
        if targets.size:
            assert targets.min() >= 1
            assert targets.max() <= 40


def test_no_target_equals_zero() -> None:
    """0 is the CTC blank/pad value -- it must never appear in a real target."""
    for trial_id in list(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))[:50]:
        _, targets = read_trial(DATA_ROOT, trial_id)
        assert not np.any(targets == 0)


def test_n_time_steps_at_least_seq_len_for_100_trials() -> None:
    """CTC's hard requirement: input length must not be shorter than target length."""
    ids = list(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))[:100]
    for trial_id in ids:
        assert trial_id.n_time_steps >= trial_id.seq_len


def test_reading_same_trial_twice_returns_identical_arrays() -> None:
    trial_id = next(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))
    features_a, targets_a = read_trial(DATA_ROOT, trial_id)
    features_b, targets_b = read_trial(DATA_ROOT, trial_id)
    assert np.array_equal(features_a, features_b)
    assert np.array_equal(targets_a, targets_b)


def test_read_trials_returns_results_in_input_order() -> None:
    ids = list(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))[:10]
    # Reversed so that order-preservation isn't trivially satisfied by an
    # implementation that happens to process trials in file order.
    shuffled_ids = list(reversed(ids))

    results = read_trials(DATA_ROOT, shuffled_ids)

    assert len(results) == len(shuffled_ids)
    for trial_id, (features, targets) in zip(shuffled_ids, results):
        expected_features, expected_targets = read_trial(DATA_ROOT, trial_id)
        assert np.array_equal(features, expected_features)
        assert np.array_equal(targets, expected_targets)


def test_read_trial_raises_on_test_partition_trial() -> None:
    test_trial_id = next(iter_trial_ids(DATA_ROOT, "t15.2023.08.13", partitions=("test",)))
    assert not test_trial_id.has_labels

    with pytest.raises(ValueError):
        read_trial(DATA_ROOT, test_trial_id)


def test_validate_true_raises_on_corrupted_trial_id() -> None:
    trial_id = next(iter_trial_ids(DATA_ROOT, "t15.2023.08.13", partitions=("train",)))
    # Larger than the real n_time_steps: the HDF5 slice truncates to the
    # actual array length, so features.shape[0] disagrees with the
    # (wrong) n_time_steps on the corrupted TrialId, which validate=True
    # must catch.
    corrupted = dataclasses.replace(trial_id, n_time_steps=trial_id.n_time_steps + 1000)

    with pytest.raises(ValueError):
        read_trial(DATA_ROOT, corrupted, validate=True)


def test_trial_id_is_hashable_and_usable_as_dict_key() -> None:
    trial_id = next(iter_trial_ids(DATA_ROOT, "t15.2023.08.11"))
    mapping = {trial_id: "value"}
    assert mapping[trial_id] == "value"
    assert trial_id in {trial_id}


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason=f"manifest not present at {MANIFEST_PATH}")
def test_trial_counts_match_manifest_for_3_sampled_sessions() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    sampled_sessions = ["t15.2023.08.11", "t15.2024.03.15", "t15.2025.01.12"]

    for session_id in sampled_sessions:
        ids = list(iter_trial_ids(DATA_ROOT, session_id, partitions=("train", "val")))
        expected = manifest[
            (manifest["session_id"] == session_id) & manifest["partition"].isin(["train", "val"])
        ]["n_trials"].sum()
        assert len(ids) == expected, f"{session_id}: got {len(ids)}, manifest says {expected}"
