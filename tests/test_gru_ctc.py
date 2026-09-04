"""Unit tests for `b2t_drift.models.gru_ctc.GRUCTC`, synthetic data only.

Targets the fixes from the "Six tasks" review: padding masking around the
GRU (Task 1a), the batch-composition-dependence of Gaussian smoothing
(Task 1b, quantified rather than just asserted), the smoothing-buffer
registration inconsistency (Task 1c), the input-shape assertion (Task 1d),
and the trainable-vs-all parameter count distinction (Task 1e).
"""

from __future__ import annotations

import math

import pytest
import torch

from b2t_drift.models.gru_ctc import GRUCTC


def _tiny_model(**overrides) -> GRUCTC:
    kwargs = {"n_sessions": 1, "n_layers": 1, "hidden_size": 16, "use_day_layers": False}
    kwargs.update(overrides)
    return GRUCTC(**kwargs)


# --- Task 1(a): padding masking / bidirectional refusal --------------------


def test_bidirectional_true_raises_in_init() -> None:
    with pytest.raises(ValueError, match="bidirectional"):
        GRUCTC(n_sessions=1, bidirectional=True)


def test_forward_with_input_lengths_matches_forward_without_for_unidirectional() -> None:
    """Causal GRU: packing around padding must not change valid-timestep
    outputs, only skip computing (never using) the padded ones."""
    torch.manual_seed(0)
    model = _tiny_model(smoothing_sigma_bins=0.0)  # isolate the GRU packing effect
    model.eval()

    short_len = 12
    long_len = 40
    batch = torch.zeros(2, long_len, 512)
    batch[0, :short_len] = torch.randn(short_len, 512)
    batch[1] = torch.randn(long_len, 512)
    session_indices = torch.zeros(2, dtype=torch.int64)
    input_lengths = torch.tensor([short_len, long_len], dtype=torch.int64)

    with torch.no_grad():
        packed = model(batch, session_indices, input_lengths=input_lengths)
        unpacked = model(batch, session_indices)

    assert packed.shape == unpacked.shape == (long_len, 2, model.n_classes)
    # Valid region of the short trial must match regardless of packing.
    assert torch.allclose(
        packed[:short_len, 0, :], unpacked[:short_len, 0, :], atol=1e-5,
    )
    # The long trial has no padding at all -- packing changes nothing.
    assert torch.allclose(packed[:, 1, :], unpacked[:, 1, :], atol=1e-5)


def test_forward_input_lengths_zeroes_padded_region_after_unpacking() -> None:
    """`pad_packed_sequence` zero-fills the GRU's OWN output past each
    item's real length -- but `forward` applies the linear output layer
    and log_softmax AFTER that, so the padded region of the returned
    log_probs is not literally all-zero (log_softmax of an all-zero
    logit vector is uniform, not zero). What IS exactly checkable: with
    gru_out=0, Linear(0) = bias, so the padded region must equal
    log_softmax(output.bias), constant across every padded timestep."""
    torch.manual_seed(0)
    model = _tiny_model(smoothing_sigma_bins=0.0)
    model.eval()

    short_len, long_len = 5, 20
    batch = torch.randn(2, long_len, 512)
    session_indices = torch.zeros(2, dtype=torch.int64)
    input_lengths = torch.tensor([short_len, long_len], dtype=torch.int64)

    with torch.no_grad():
        packed = model(batch, session_indices, input_lengths=input_lengths)
        expected_padded = torch.nn.functional.log_softmax(model.output.bias, dim=-1)

    padded_region = packed[short_len:, 0, :]
    assert torch.allclose(padded_region, expected_padded.expand_as(padded_region), atol=1e-5)


# --- Task 1(b): smoothing batch-composition sensitivity ---------------------


def _batch_composition_diff(model: GRUCTC, short_len: int = 30, long_len: int = 120) -> float:
    """Max abs log_probs difference for one trial decoded alone vs. padded
    inside a larger batch, over the trial's own valid timesteps."""
    torch.manual_seed(1)
    trial = torch.randn(1, short_len, 512)
    session_idx = torch.zeros(1, dtype=torch.int64)

    with torch.no_grad():
        alone = model(trial, session_idx)

        padded_batch = torch.zeros(2, long_len, 512)
        padded_batch[0, :short_len] = trial[0]
        padded_batch[1] = torch.randn(long_len, 512)
        batched = model(padded_batch, torch.zeros(2, dtype=torch.int64))

    alone_valid = alone[:, 0, :]
    batched_valid = batched[:short_len, 0, :]
    return (alone_valid - batched_valid).abs().max().item()


def test_smoothing_batch_composition_sensitivity_reports_magnitude(capsys) -> None:
    """Quantifies the documented `_smooth` limitation, for BOTH cases its
    corrected docstring distinguishes:

    - A FRESHLY CONSTRUCTED model (DayLayers still at its identity
      init): the collate-padding zeros and F.conv1d's own implicit edge
      zero-padding are mathematically indistinguishable, so this should
      measure ~0 (verified directly, not assumed -- see _smooth's
      docstring for why an earlier, vaguer version of this claim was
      imprecise about the mechanism).
    - A model whose DayLayers has been perturbed away from identity
      (simulating a post-training state, via a nonzero bias): now the
      collate-padding gets mapped to that nonzero bias BEFORE smoothing
      sees it, which genuinely differs from conv1d's own zero
      edge-padding in the alone case -- this should be clearly nonzero.

    Reports both magnitudes rather than gating on a tight threshold --
    the second case is EXPECTED to be nonzero, per _smooth's docstring;
    the point is visibility, not a pass/fail bar on a known, accepted
    effect.
    """
    torch.manual_seed(0)
    fresh_model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=16, use_day_layers=True)
    fresh_model.eval()
    fresh_diff = _batch_composition_diff(fresh_model)

    torch.manual_seed(0)
    trained_model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=16, use_day_layers=True)
    trained_model.eval()
    with torch.no_grad():
        trained_model.day_layers.biases[0] = torch.randn(512) * 0.5
    trained_diff = _batch_composition_diff(trained_model)

    print("\n[Task 1(b)] max abs log_probs difference, trial alone vs. "
          "padded in a larger batch:")
    print(f"[Task 1(b)]   fresh (identity-init) DayLayers:  {fresh_diff:.8f}")
    print(f"[Task 1(b)]   trained (nonzero-bias) DayLayers: {trained_diff:.8f}")
    if trained_diff > 1e-3:
        print("[Task 1(b)] the trained case exceeds 1e-3 -- this is the "
              "documented, accepted batch-composition-dependence of "
              "smoothing (once DayLayers has moved off identity), not a bug.")

    captured = capsys.readouterr()
    assert "max abs log_probs difference" in captured.out
    assert math.isfinite(fresh_diff)
    assert math.isfinite(trained_diff)
    # The corrected docstring's central claim, made checkable: a fresh
    # model shows (near-)zero batch-dependence; a trained-bias one does not.
    assert fresh_diff < 1e-4
    assert trained_diff > fresh_diff


# --- Task 1(c): smoothing buffer registration consistency -------------------


def test_smoothing_buffer_registered_consistently_with_and_without_smoothing() -> None:
    with_smoothing = _tiny_model(smoothing_sigma_bins=2.0)
    without_smoothing = _tiny_model(smoothing_sigma_bins=0.0)

    # _smoothing_kernel is a registered buffer NAME either way, even
    # though PyTorch's state_dict()/named_buffers() skip a None-valued
    # buffer's VALUE -- see __init__'s comment for why this narrower
    # claim is what's actually true and achievable here.
    assert "_smoothing_kernel" in with_smoothing._buffers
    assert "_smoothing_kernel" in without_smoothing._buffers

    with_keys = set(with_smoothing.state_dict().keys())
    without_keys = set(without_smoothing.state_dict().keys())
    assert with_keys - without_keys == {"_smoothing_kernel"}
    assert without_keys - with_keys == set()


# --- Task 1(d): input shape assertion ---------------------------------------


def test_forward_raises_on_wrong_feature_dimension() -> None:
    model = _tiny_model()
    wrong = torch.randn(2, 10, 256)  # transposed / wrong last dim
    session_indices = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match=r"512.*256|256.*512"):
        model(wrong, session_indices)


# --- Task 1(e): n_parameters() vs n_trainable_parameters() -----------------


def test_n_parameters_counts_all_regardless_of_requires_grad() -> None:
    model = GRUCTC(n_sessions=2, n_layers=1, hidden_size=16, use_day_layers=True)
    for p in model.day_layers.parameters():
        p.requires_grad_(False)

    all_counts = model.n_parameters()
    trainable_counts = model.n_trainable_parameters()

    assert all_counts["day_layers"] > 0
    assert trainable_counts["day_layers"] == 0
    assert all_counts["total"] > trainable_counts["total"]
    # gru/output were untouched -- both views agree there.
    assert all_counts["gru"] == trainable_counts["gru"]
    assert all_counts["output"] == trainable_counts["output"]


def test_n_trainable_parameters_matches_n_parameters_when_nothing_frozen() -> None:
    model = GRUCTC(n_sessions=2, n_layers=1, hidden_size=16, use_day_layers=True)
    assert model.n_parameters() == model.n_trainable_parameters()
