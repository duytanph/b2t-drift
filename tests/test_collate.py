"""Tests for b2t_drift.data.collate (synthetic data only -- this module
never touches HDF5, so these tests don't need the real dataset present).
"""

from __future__ import annotations

from datetime import date

import pytest
import torch

from b2t_drift.data.collate import LengthBucketedSampler, collate_trials
from b2t_drift.data.hdf5_io import TrialId


def make_item(
    n_time_steps: int,
    seq_len: int,
    session_id: str = "t15.2023.08.11",
    session_index: int = 0,
    block_num: int = 1,
    trial_key: str = "trial_0000",
    target_seed: int = 0,
) -> dict:
    """Build one synthetic TrialDataset.__getitem__-shaped item."""
    generator = torch.Generator().manual_seed(target_seed)
    features = torch.randn(n_time_steps, 512, generator=generator)
    targets = torch.randint(1, 41, (seq_len,), generator=generator, dtype=torch.int64)
    return {
        "features": features,
        "targets": targets,
        "n_time_steps": n_time_steps,
        "seq_len": seq_len,
        "session_index": session_index,
        "session_id": session_id,
        "block_num": block_num,
        "trial_key": trial_key,
    }


def make_trial_id(n_time_steps: int, seq_len: int = 10, trial_key: str = "trial_0000") -> TrialId:
    return TrialId(
        session_id="t15.2023.08.11",
        session_date=date(2023, 8, 11),
        partition="train",
        trial_key=trial_key,
        block_num=1,
        trial_num=0,
        n_time_steps=n_time_steps,
        seq_len=seq_len,
        sentence_label="synthetic",
    )


# --- collate_trials -----------------------------------------------------

MIXED_BATCH = [
    make_item(100, 5, trial_key="trial_a", target_seed=1),
    make_item(250, 12, trial_key="trial_b", target_seed=2),
    make_item(50, 3, trial_key="trial_c", target_seed=3),
    make_item(180, 8, trial_key="trial_d", target_seed=4),
]


def test_batch_shapes_correct_for_mixed_length_batch() -> None:
    batch = collate_trials(MIXED_BATCH)
    max_time = max(item["n_time_steps"] for item in MIXED_BATCH)
    total_targets = sum(item["seq_len"] for item in MIXED_BATCH)

    assert batch["features"].shape == (4, max_time, 512)
    assert batch["targets"].shape == (total_targets,)
    assert batch["input_lengths"].shape == (4,)
    assert batch["target_lengths"].shape == (4,)
    assert batch["session_indices"].shape == (4,)
    assert len(batch["session_ids"]) == 4
    assert len(batch["block_nums"]) == 4
    assert len(batch["trial_keys"]) == 4


def test_features_padded_region_is_exactly_zero() -> None:
    batch = collate_trials(MIXED_BATCH)
    max_time = batch["features"].shape[1]
    for i, item in enumerate(MIXED_BATCH):
        t = item["n_time_steps"]
        if t < max_time:
            pad_region = batch["features"][i, t:, :]
            assert torch.all(pad_region == 0.0)
        # And the real region should match the original (not zeroed).
        assert torch.equal(batch["features"][i, :t, :], item["features"])


def test_input_lengths_matches_n_time_steps() -> None:
    batch = collate_trials(MIXED_BATCH)
    expected = torch.tensor([item["n_time_steps"] for item in MIXED_BATCH], dtype=torch.int64)
    assert torch.equal(batch["input_lengths"], expected)


def test_target_lengths_matches_seq_len() -> None:
    batch = collate_trials(MIXED_BATCH)
    expected = torch.tensor([item["seq_len"] for item in MIXED_BATCH], dtype=torch.int64)
    assert torch.equal(batch["target_lengths"], expected)


def test_targets_is_1d_and_length_matches_target_lengths_sum() -> None:
    batch = collate_trials(MIXED_BATCH)
    assert batch["targets"].dim() == 1
    assert batch["targets"].shape[0] == int(batch["target_lengths"].sum())


def test_concatenated_targets_round_trip_by_target_lengths() -> None:
    batch = collate_trials(MIXED_BATCH)
    lengths = batch["target_lengths"].tolist()

    offset = 0
    for item, length in zip(MIXED_BATCH, lengths):
        recovered = batch["targets"][offset : offset + length]
        assert torch.equal(recovered, item["targets"])
        offset += length


def test_value_error_when_n_time_steps_less_than_seq_len() -> None:
    bad_item = make_item(5, 10, trial_key="trial_bad")  # n_time_steps < seq_len
    with pytest.raises(ValueError, match="trial_bad"):
        collate_trials([bad_item, MIXED_BATCH[0]])


def test_value_error_when_target_contains_zero() -> None:
    item = make_item(50, 5, trial_key="trial_zero")
    item["targets"] = torch.tensor([1, 2, 0, 3, 4], dtype=torch.int64)  # 0 = blank, invalid
    with pytest.raises(ValueError, match="trial_zero"):
        collate_trials([item])


def test_dtypes() -> None:
    batch = collate_trials(MIXED_BATCH)
    assert batch["features"].dtype == torch.float32
    assert batch["targets"].dtype == torch.int64
    assert batch["input_lengths"].dtype == torch.int64
    assert batch["target_lengths"].dtype == torch.int64
    assert batch["session_indices"].dtype == torch.int64


def test_end_to_end_ctc_smoke_test() -> None:
    """Catches length-argument-order errors, which are otherwise silent."""
    batch = collate_trials(MIXED_BATCH)
    max_time = batch["features"].shape[1]
    batch_size = batch["features"].shape[0]
    n_classes = 41

    logits = torch.randn(max_time, batch_size, n_classes, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(
        log_probs,
        batch["targets"],
        batch["input_lengths"],
        batch["target_lengths"],
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()  # also exercises the length args through autograd


# --- LengthBucketedSampler -----------------------------------------------

def make_trial_ids_with_lengths(lengths: list[int]) -> list[TrialId]:
    return [make_trial_id(n, trial_key=f"trial_{i:04d}") for i, n in enumerate(lengths)]


def test_sampler_yields_every_index_exactly_once_per_epoch() -> None:
    lengths = [100, 500, 250, 1200, 90, 300, 800, 150, 2000, 60, 400, 700]
    trial_ids = make_trial_ids_with_lengths(lengths)
    sampler = LengthBucketedSampler(trial_ids, batch_size=4, shuffle=True, seed=42)

    seen = []
    for batch in sampler:
        seen.extend(batch)

    assert sorted(seen) == list(range(len(trial_ids)))


def test_sampler_reduces_padding_waste_vs_random_batching() -> None:
    import random as _random

    lengths = [
        50, 60, 2000, 1900, 55, 65, 1800, 1950,
        70, 45, 2100, 1700, 58, 62, 2050, 1850,
    ]
    trial_ids = make_trial_ids_with_lengths(lengths)
    batch_size = 4

    def padding_waste(batches: list[list[int]]) -> int:
        total = 0
        for batch in batches:
            batch_lengths = [lengths[i] for i in batch]
            total += (max(batch_lengths) * len(batch_lengths)) - sum(batch_lengths)
        return total

    sampler = LengthBucketedSampler(trial_ids, batch_size=batch_size, shuffle=False, bucket_noise=0.0)
    bucketed_batches = list(sampler)
    bucketed_waste = padding_waste(bucketed_batches)

    rng = _random.Random(0)
    indices = list(range(len(trial_ids)))
    rng.shuffle(indices)
    random_batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    random_waste = padding_waste(random_batches)

    assert bucketed_waste < random_waste


def test_sampler_bucket_noise_changes_composition_across_epochs() -> None:
    # Closely-spaced lengths so noise can plausibly reorder neighbors.
    lengths = list(range(100, 100 + 40))  # 100..139, adjacent gaps of 1
    trial_ids = make_trial_ids_with_lengths(lengths)
    sampler = LengthBucketedSampler(
        trial_ids, batch_size=4, shuffle=False, seed=0, bucket_noise=0.15,
    )

    sampler.set_epoch(0)
    epoch0_batches = [tuple(sorted(b)) for b in sampler]

    sampler.set_epoch(1)
    epoch1_batches = [tuple(sorted(b)) for b in sampler]

    assert epoch0_batches != epoch1_batches


def test_iter_without_set_epoch_is_pure_and_repeatable() -> None:
    """__iter__ must not mutate state -- two calls with no intervening
    set_epoch() yield identical batches; set_epoch() is the only thing
    that changes future iteration."""
    lengths = list(range(100, 100 + 40))
    trial_ids = make_trial_ids_with_lengths(lengths)
    sampler = LengthBucketedSampler(
        trial_ids, batch_size=4, shuffle=True, seed=0, bucket_noise=0.15,
    )

    first_call = list(sampler)
    second_call = list(sampler)
    assert first_call == second_call

    sampler.set_epoch(1)
    third_call = list(sampler)
    assert third_call != first_call


def test_drop_last_false_includes_short_final_batch() -> None:
    lengths = list(range(10))  # 10 trials, batch_size 4 -> batches of 4,4,2
    trial_ids = make_trial_ids_with_lengths(lengths)
    sampler = LengthBucketedSampler(trial_ids, batch_size=4, shuffle=False, drop_last=False)

    batches = list(sampler)
    assert len(sampler) == 3
    assert len(batches) == 3
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(i for b in batches for i in b) == list(range(10))


def test_drop_last_true_drops_short_final_batch() -> None:
    lengths = list(range(10))
    trial_ids = make_trial_ids_with_lengths(lengths)
    sampler = LengthBucketedSampler(trial_ids, batch_size=4, shuffle=False, drop_last=True)

    batches = list(sampler)
    assert len(sampler) == 2
    assert len(batches) == 2
    assert [len(b) for b in batches] == [4, 4]
    # The 2 trials in the dropped short batch are legitimately absent.
    assert len(sorted(i for b in batches for i in b)) == 8


def test_drop_last_with_3969_trials_batch_size_32() -> None:
    """The exact TRAIN block size (docs/DATASET.md section 13) and batch
    size that makes the last batch of an un-dropped epoch a single trial."""
    n_trials = 3969
    batch_size = 32
    lengths = [100 + (i % 50) for i in range(n_trials)]  # arbitrary lengths
    trial_ids = make_trial_ids_with_lengths(lengths)

    with_drop = LengthBucketedSampler(trial_ids, batch_size=batch_size, shuffle=False, drop_last=True)
    without_drop = LengthBucketedSampler(trial_ids, batch_size=batch_size, shuffle=False, drop_last=False)

    assert len(with_drop) == 124
    assert len(without_drop) == 125

    with_drop_batches = list(with_drop)
    without_drop_batches = list(without_drop)
    assert len(with_drop_batches) == 124
    assert len(without_drop_batches) == 125
    assert all(len(b) == batch_size for b in with_drop_batches)
    assert [len(b) for b in without_drop_batches[:-1]] == [batch_size] * 124
    assert len(without_drop_batches[-1]) == 1  # the leftover trial

    # drop_last=False must still cover every trial exactly once.
    seen = sorted(i for b in without_drop_batches for i in b)
    assert seen == list(range(n_trials))
