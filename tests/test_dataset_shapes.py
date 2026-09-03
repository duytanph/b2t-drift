"""Tests that TrialDataset examples have the expected tensor shapes.

`test_example_shapes_match_config` is implemented against the real T15
data, now that `TrialDataset` exists (see tests/test_dataset.py for the
fuller test suite). `test_collated_batch_shapes` stays a stub -- collate.py
does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="stub: module not yet implemented")
def test_collated_batch_shapes() -> None:
    """A collated batch should pad to (batch, max_time, num_features)."""
    raise NotImplementedError
