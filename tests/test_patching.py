"""Tests for GRUCTC's input patching (patch_size/patch_stride) and the
matching ctc_input_lengths formula.

THIS IS THE HIGHEST-RISK TEST FILE IN THE MODEL LAYER. Getting the
length-conversion formula wrong here produces a model that trains,
converges, and emits garbage with no error raised (`CLAUDE.md`'s "known
pitfalls") -- these tests exist specifically to catch that class of bug
before it reaches a real training run.

Synthetic tests (length arithmetic, no-op equivalence) always run.
Real-data tests (the CTC length-constraint sweep, the patched overfit
gate) are skipped if `data/raw/hdf5_data_final` is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from b2t_drift.data.collate import collate_trials
from b2t_drift.data.hdf5_io import iter_trial_ids, read_trial
from b2t_drift.data.splits import load_split, split_to_trial_ids
from b2t_drift.evaluation.ctc_decode import greedy_ctc_decode
from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.phonemes import BLANK_INDEX, N_CLASSES

DATA_ROOT = Path("data/raw/hdf5_data_final")
SPLIT_PATH = Path("splits/temporal_s01_s15_switchboard.json")
SESSION_ID = "t15.2023.08.11"

PATCH_SIZE = 14
PATCH_STRIDE = 4


def _require_real_data() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")


# --- ctc_input_lengths matches the actual patched time dimension -----------


@pytest.mark.parametrize(
    "input_len",
    [14, 15, 17, 18, 20, 21, 25, 50, 137, 138, 141, 527, 528, 1000, 1001],
)
def test_ctc_input_lengths_matches_actual_patched_time_dimension(input_len: int) -> None:
    model = GRUCTC(
        n_sessions=1, n_layers=1, hidden_size=8, use_day_layers=False,
        smoothing_sigma_bins=0.0, patch_size=PATCH_SIZE, patch_stride=PATCH_STRIDE,
    )
    x = torch.randn(1, input_len, 512)
    patched = model._patch(x)

    expected_n_patches = (input_len - PATCH_SIZE) // PATCH_STRIDE + 1
    assert patched.shape[1] == expected_n_patches, (
        f"input_len={input_len}: _patch produced {patched.shape[1]} patches, "
        f"expected {expected_n_patches}"
    )

    computed = model.ctc_input_lengths(torch.tensor([input_len], dtype=torch.int64))
    assert computed.item() == expected_n_patches, (
        f"input_len={input_len}: ctc_input_lengths returned {computed.item()}, "
        f"expected {expected_n_patches} (must match _patch's actual output exactly)"
    )


def test_patch_concatenation_is_bin_major_not_feature_major() -> None:
    """Directly verifies the concatenation ORDER: patch position 0's full
    feature vector, then position 1's, ... -- not feature-major (all
    patch_size values of feature 0, then feature 1, ...). Uses distinct,
    recognizable per-timestep constants so the two orderings are
    trivially distinguishable."""
    model = GRUCTC(
        n_sessions=1, n_layers=1, hidden_size=4, use_day_layers=False,
        smoothing_sigma_bins=0.0, patch_size=3, patch_stride=3,
    )
    n_features = 512
    x = torch.zeros(1, 3, n_features)
    x[0, 0, :] = 10.0  # bin 0: all features = 10
    x[0, 1, :] = 20.0  # bin 1: all features = 20
    x[0, 2, :] = 30.0  # bin 2: all features = 30

    patched = model._patch(x)  # (1, 1, 3*512)
    assert patched.shape == (1, 1, 3 * n_features)
    # Bin-major: first n_features entries are bin 0 (all 10), next
    # n_features are bin 1 (all 20), last n_features are bin 2 (all 30).
    assert torch.all(patched[0, 0, :n_features] == 10.0)
    assert torch.all(patched[0, 0, n_features:2 * n_features] == 20.0)
    assert torch.all(patched[0, 0, 2 * n_features:] == 30.0)


# --- patch_size=1, patch_stride=1 is a no-op --------------------------------


def test_patch_and_ctc_input_lengths_are_literal_noops_at_defaults() -> None:
    model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=8, use_day_layers=False)
    assert model.patch_size == 1
    assert model.patch_stride == 1

    x = torch.randn(2, 30, 512)
    assert model._patch(x) is x  # same tensor object, not just equal values

    lengths = torch.tensor([10, 30], dtype=torch.int64)
    assert model.ctc_input_lengths(lengths) is lengths


def test_default_patching_matches_hand_replicated_pre_patching_forward() -> None:
    """patch_size=1, patch_stride=1 (the default) must reproduce EXACTLY
    what forward() computes with no patching step at all -- verified
    against an independent reference that hand-replicates the pipeline
    (mask -> smooth -> pack with RAW input_lengths, no _patch, no
    ctc_input_lengths detour), not just by checking shapes match."""
    torch.manual_seed(0)
    model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=16, use_day_layers=True, smoothing_sigma_bins=2.0)
    model.eval()

    x = torch.randn(2, 30, 512)
    session_indices = torch.zeros(2, dtype=torch.int64)
    input_lengths = torch.tensor([20, 30], dtype=torch.int64)

    with torch.no_grad():
        actual = model(x, session_indices, input_lengths=input_lengths)

        ref = x
        if model.day_layers is not None:
            ref = model.day_layers(ref, session_indices)
        time_indices = torch.arange(ref.size(1))
        valid_mask = time_indices.unsqueeze(0) < input_lengths.unsqueeze(1)
        ref = ref * valid_mask.unsqueeze(-1)
        ref = model._smooth(ref)
        packed = nn.utils.rnn.pack_padded_sequence(
            ref, input_lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = model.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=ref.size(1),
        )
        logits = model.output(gru_out)
        expected = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

    assert torch.allclose(actual, expected, atol=1e-5)


# --- CTC constraint holds after patching, on real s15 train+val trials -----


def test_ctc_length_constraint_holds_after_patching_for_s15_train_and_val() -> None:
    """ctc_len >= target_len (CTC's hard input>=target requirement) for
    EVERY trial in the s15 train+val partitions, at patch_size=14,
    patch_stride=4. Median input:target ratio is ~33:1 (docs/DATASET.md
    section 6), so 4x downsampling should leave roughly 8:1, comfortably
    above the 1:1 floor -- verified directly rather than assumed, and any
    failing trial is named rather than just counted."""
    _require_real_data()
    if not SPLIT_PATH.exists():
        pytest.skip(f"split manifest not present at {SPLIT_PATH}")

    split = load_split(SPLIT_PATH)
    train_ids = split_to_trial_ids(split, DATA_ROOT, "train")
    val_ids = split_to_trial_ids(split, DATA_ROOT, "val")

    failures = []
    ratios = []
    for trial_id in [*train_ids, *val_ids]:
        ctc_len = (trial_id.n_time_steps - PATCH_SIZE) // PATCH_STRIDE + 1
        ratios.append(ctc_len / trial_id.seq_len if trial_id.seq_len else float("inf"))
        if ctc_len < trial_id.seq_len:
            failures.append(
                (trial_id.session_id, trial_id.trial_key, trial_id.n_time_steps, trial_id.seq_len, ctc_len)
            )

    assert not failures, (
        f"{len(failures)} of {len(train_ids) + len(val_ids)} trial(s) violate "
        f"ctc_len >= target_len after patch_size={PATCH_SIZE}/patch_stride={PATCH_STRIDE}: "
        f"{failures[:10]}"
    )
    # Sanity check on the ratio claim in this test's own docstring.
    median_ratio = sorted(ratios)[len(ratios) // 2]
    assert median_ratio > 4, f"median post-patch ctc_len:target_len ratio {median_ratio:.2f} is not comfortably above 1:1"


# --- end-to-end overfit gate with patching enabled --------------------------

N_STEPS = 300
LOSS_ABSOLUTE_THRESHOLD = 0.5


def _load_two_items() -> list[dict]:
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:2]
    assert len(trial_ids) == 2, f"expected >=2 trials in {SESSION_ID}, found {len(trial_ids)}"
    items = []
    for i, trial_id in enumerate(trial_ids):
        features, targets = read_trial(DATA_ROOT, trial_id)
        items.append({
            "features": torch.from_numpy(features).float(),
            "targets": torch.from_numpy(targets).long(),
            "n_time_steps": trial_id.n_time_steps,
            "seq_len": trial_id.seq_len,
            "session_index": i,
            "session_id": trial_id.session_id,
            "block_num": trial_id.block_num,
            "trial_key": trial_id.trial_key,
        })
    return items


def test_overfit_two_real_trials_with_patching_enabled() -> None:
    """THE PATCHING GATE (mirrors test_overfit_one_batch.py's pipeline
    gate): with patch_size=14, patch_stride=4 -- the exact configuration
    Task 3 trains with -- the model must still be able to memorize two
    real trials, and its greedy decode must recover the correct phoneme
    sequence. This is the test that would catch a length-arithmetic
    error: get ctc_input_lengths wrong and the model either fails to
    memorize at all, or "succeeds" while decoding garbage (CLAUDE.md's
    "known pitfalls")."""
    _require_real_data()
    torch.manual_seed(0)

    items = _load_two_items()
    batch = collate_trials(items)
    model = GRUCTC(
        n_sessions=2, hidden_size=256, n_layers=2, dropout=0.0, use_day_layers=True,
        patch_size=PATCH_SIZE, patch_stride=PATCH_STRIDE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=False)

    losses = []
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        log_probs = model(
            batch["features"], batch["session_indices"], input_lengths=batch["input_lengths"],
        )
        loss = ctc_loss_fn(
            log_probs,
            batch["targets"],
            model.ctc_input_lengths(batch["input_lengths"]),
            batch["target_lengths"],
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final = losses[-1]
    assert final < LOSS_ABSOLUTE_THRESHOLD, (
        f"final loss {final:.4f} did not drop below {LOSS_ABSOLUTE_THRESHOLD} with "
        f"patching enabled -- likely a length-arithmetic error, not just slow convergence."
    )

    model.eval()
    with torch.no_grad():
        log_probs = model(
            batch["features"], batch["session_indices"], input_lengths=batch["input_lengths"],
        )
    patched_lengths = model.ctc_input_lengths(batch["input_lengths"])
    decoded = greedy_ctc_decode(
        log_probs.transpose(0, 1), lengths=patched_lengths, blank_index=BLANK_INDEX, n_classes=N_CLASSES,
    )
    true_targets = [item["targets"].tolist() for item in items]
    assert any(d == t for d, t in zip(decoded, true_targets)), (
        f"no trial's greedy decode matched its true phoneme sequence with patching enabled.\n"
        f"decoded: {decoded}\ntrue:    {true_targets}"
    )
