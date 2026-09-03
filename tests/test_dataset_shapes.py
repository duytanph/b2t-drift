"""Tests that TrialDataset examples and collated batches have the expected
tensor shapes.

Both tests are implemented against the real T15 data, now that
`TrialDataset` and `collate_trials` both exist (see tests/test_dataset.py
and tests/test_collate.py for the fuller test suites of each).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from b2t_drift.data.collate import collate_trials
from b2t_drift.data.dataset import TrialDataset
from b2t_drift.data.hdf5_io import iter_trial_ids
from b2t_drift.phonemes import N_CLASSES

DATA_ROOT = Path("data/raw/hdf5_data_final")


def test_example_shapes_match_config() -> None:
    """A dataset example's neural tensor should be (time, num_features)."""
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")

    session_id = "t15.2023.08.11"
    trial_ids = list(iter_trial_ids(DATA_ROOT, session_id))[:5]
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index={session_id: 0})

    for idx, trial_id in enumerate(trial_ids):
        item = dataset[idx]
        assert item["features"].shape == (trial_id.n_time_steps, 512)
        assert item["targets"].shape == (trial_id.seq_len,)
        # Sanity check against the phoneme vocabulary size (41 = blank +
        # 40 classes): every target value must be a valid class index.
        if item["targets"].numel():
            assert item["targets"].max().item() < N_CLASSES


def test_collated_batch_shapes() -> None:
    """A collated batch should pad to (batch, max_time, num_features)."""
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")

    session_id = "t15.2023.08.11"
    trial_ids = list(iter_trial_ids(DATA_ROOT, session_id))[:8]
    dataset = TrialDataset(DATA_ROOT, trial_ids, session_to_index={session_id: 0})

    items = [dataset[i] for i in range(len(trial_ids))]
    batch = collate_trials(items)

    max_time = max(t.n_time_steps for t in trial_ids)
    total_targets = sum(t.seq_len for t in trial_ids)
    assert batch["features"].shape == (len(trial_ids), max_time, 512)
    assert batch["targets"].shape == (total_targets,)
    assert batch["input_lengths"].shape == (len(trial_ids),)
    assert batch["target_lengths"].shape == (len(trial_ids),)
