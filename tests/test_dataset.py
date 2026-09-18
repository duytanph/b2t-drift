"""Tests for b2t_drift.data.dataset.TrialDataset, against the real T15 data.

The whole module is skipped if data/raw/hdf5_data_final is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from b2t_drift.data.dataset import FEATURE_SUBSETS, TrialDataset
from b2t_drift.data.hdf5_io import iter_trial_ids

DATA_ROOT = Path("data/raw/hdf5_data_final")
SESSION_ID = "t15.2023.08.11"  # train-only session, 288 labeled trials


def _take_first(batch: list) -> dict:
    """collate_fn for batch_size=1 DataLoader tests below.

    Must be a module-level function, not a lambda or closure: on Windows,
    DataLoader worker processes are started via `spawn`, which pickles the
    collate_fn to send to each worker -- a lambda or a function defined
    inside a test isn't picklable, and pickling it raises before any
    worker even starts.
    """
    return batch[0]


@pytest.fixture(autouse=True)
def _require_real_data() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")


@pytest.fixture
def trial_ids():
    return list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:30]


@pytest.fixture
def session_to_index():
    return {SESSION_ID: 0}


def test_len_equals_number_of_trial_ids(trial_ids, session_to_index) -> None:
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    assert len(dataset) == len(trial_ids)


def test_getitem_returns_all_8_keys_with_correct_types(trial_ids, session_to_index) -> None:
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    item = dataset[0]

    assert set(item.keys()) == {
        "features", "targets", "n_time_steps", "seq_len",
        "session_index", "session_id", "block_num", "trial_key",
    }
    assert isinstance(item["features"], torch.Tensor)
    assert isinstance(item["targets"], torch.Tensor)
    assert isinstance(item["n_time_steps"], int)
    assert isinstance(item["seq_len"], int)
    assert isinstance(item["session_index"], int)
    assert isinstance(item["session_id"], str)
    assert isinstance(item["block_num"], int)
    assert isinstance(item["trial_key"], str)


def test_features_shape_and_dtype_match_trial_id(trial_ids, session_to_index) -> None:
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    for idx, trial_id in enumerate(trial_ids):
        item = dataset[idx]
        assert item["features"].dtype == torch.float32
        assert item["features"].shape == (trial_id.n_time_steps, 512)


def test_feature_subset_tx_selects_first_256_columns(trial_ids, session_to_index) -> None:
    full = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    tx_only = TrialDataset(DATA_ROOT, trial_ids, session_to_index, feature_subset=FEATURE_SUBSETS["tx"])
    for idx in range(len(trial_ids)):
        full_item = full[idx]
        tx_item = tx_only[idx]
        assert tx_item["features"].shape == (full_item["features"].shape[0], 256)
        assert torch.equal(tx_item["features"], full_item["features"][:, :256])


def test_feature_subset_sbp_selects_last_256_columns(trial_ids, session_to_index) -> None:
    full = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    sbp_only = TrialDataset(DATA_ROOT, trial_ids, session_to_index, feature_subset=FEATURE_SUBSETS["sbp"])
    for idx in range(len(trial_ids)):
        full_item = full[idx]
        sbp_item = sbp_only[idx]
        assert sbp_item["features"].shape == (full_item["features"].shape[0], 256)
        assert torch.equal(sbp_item["features"], full_item["features"][:, 256:])


def test_feature_subset_all_is_a_no_op() -> None:
    assert FEATURE_SUBSETS["all"] is None


def test_targets_shape_and_dtype_match_trial_id(trial_ids, session_to_index) -> None:
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    for idx, trial_id in enumerate(trial_ids):
        item = dataset[idx]
        assert item["targets"].dtype == torch.int64
        assert item["targets"].shape == (trial_id.seq_len,)


def test_all_target_values_in_range_1_to_40(trial_ids, session_to_index) -> None:
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    for idx in range(len(dataset)):
        targets = dataset[idx]["targets"]
        if targets.numel():
            assert targets.min().item() >= 1
            assert targets.max().item() <= 40


def test_session_index_matches_provided_mapping(trial_ids) -> None:
    session_to_index = {SESSION_ID: 7}  # arbitrary, non-default value
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    item = dataset[0]
    assert item["session_index"] == 7
    assert item["session_id"] == SESSION_ID


def test_init_raises_on_session_missing_from_mapping(trial_ids) -> None:
    with pytest.raises(ValueError):
        TrialDataset(DATA_ROOT, trial_ids, session_to_index={})


def test_init_raises_on_unlabeled_trial(trial_ids, session_to_index) -> None:
    test_trial_id = next(iter_trial_ids(DATA_ROOT, "t15.2023.08.13", partitions=("test",)))
    assert not test_trial_id.has_labels

    combined_session_to_index = {**session_to_index, test_trial_id.session_id: 1}
    with pytest.raises(ValueError):
        TrialDataset(DATA_ROOT, [*trial_ids, test_trial_id], combined_session_to_index)


def test_two_reads_of_same_index_are_identical(trial_ids, session_to_index) -> None:
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    item_a = dataset[0]
    item_b = dataset[0]
    assert torch.equal(item_a["features"], item_b["features"])
    assert torch.equal(item_a["targets"], item_b["targets"])


@pytest.mark.timeout(90)
def test_dataloader_with_multiple_workers_does_not_hang_or_corrupt(session_to_index) -> None:
    """Proves the pid-keyed hdf5_io handle cache works across processes.

    If DataLoader worker processes ended up sharing an h5py handle opened
    in the parent process, this would hang or return corrupted data --
    see b2t_drift.data.hdf5_io's module docstring. A generous timeout is
    set so that failure mode fails this test rather than blocking the
    whole suite.
    """
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:24]
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=_take_first,
    )

    n_seen = 0
    for trial_id, item in zip(trial_ids, loader):
        assert item["trial_key"] == trial_id.trial_key
        assert item["features"].shape == (trial_id.n_time_steps, 512)
        assert item["targets"].shape == (trial_id.seq_len,)
        assert item["features"].dtype == torch.float32
        assert item["targets"].dtype == torch.int64
        n_seen += 1

    assert n_seen == len(trial_ids)
    assert n_seen >= 20
