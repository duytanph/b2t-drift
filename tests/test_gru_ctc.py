"""Unit tests for `b2t_drift.models.gru_ctc.GRUCTC`, synthetic data only.

Targets the fixes from the "Six tasks" review: padding masking around the
GRU (Task 1a), the batch-composition-dependence of Gaussian smoothing --
now FIXED by masking padding right after DayLayers, before smoothing
(Task 1b of the "Implement the training loop" round; quantified rather
than just asserted), the smoothing-buffer registration inconsistency
(Task 1c), the input-shape assertion (Task 1d), and the trainable-vs-all
parameter count distinction (Task 1e).
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


def test_input_lengths_none_raises() -> None:
    """input_lengths is now mandatory (not an opt-out default) -- see the
    module docstring's 'PADDING MUST BE MASKED BEFORE SMOOTHING'."""
    model = _tiny_model()
    features = torch.randn(2, 10, 512)
    session_indices = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="input_lengths"):
        model(features, session_indices)


def test_packing_matches_direct_gru_call_for_unidirectional() -> None:
    """Packing (mandatory, now that input_lengths is required) must not
    change valid-timestep outputs for a causal GRU, only skip computing
    (never using) the padded ones -- verified against an independent,
    unpacked reference that manually replicates forward()'s pre-GRU
    steps (mask, then smooth) and calls model.gru directly."""
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
        packed_log_probs = model(batch, session_indices, input_lengths=input_lengths)

        # Independent (unpacked) reference: replicate forward()'s
        # pre-GRU steps by hand, then call the GRU directly.
        x = batch
        if model.day_layers is not None:
            x = model.day_layers(x, session_indices)
        time_indices = torch.arange(x.size(1))
        valid_mask = time_indices.unsqueeze(0) < input_lengths.unsqueeze(1)
        x = x * valid_mask.unsqueeze(-1)
        x = model._smooth(x)
        gru_out_direct, _ = model.gru(x)
        logits_direct = model.output(gru_out_direct)
        log_probs_direct = torch.nn.functional.log_softmax(logits_direct, dim=-1).transpose(0, 1)

    assert packed_log_probs.shape == log_probs_direct.shape == (long_len, 2, model.n_classes)
    # Valid region of the short trial must match regardless of packing.
    assert torch.allclose(
        packed_log_probs[:short_len, 0, :], log_probs_direct[:short_len, 0, :], atol=1e-5,
    )
    # The long trial has no padding at all -- packing changes nothing.
    assert torch.allclose(packed_log_probs[:, 1, :], log_probs_direct[:, 1, :], atol=1e-5)


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
    inside a larger batch, over the trial's own valid timesteps. Passes
    input_lengths for BOTH calls (mandatory since the fix -- see the
    module docstring)."""
    torch.manual_seed(1)
    trial = torch.randn(1, short_len, 512)
    session_idx = torch.zeros(1, dtype=torch.int64)
    alone_lengths = torch.tensor([short_len], dtype=torch.int64)

    with torch.no_grad():
        alone = model(trial, session_idx, input_lengths=alone_lengths)

        padded_batch = torch.zeros(2, long_len, 512)
        padded_batch[0, :short_len] = trial[0]
        padded_batch[1] = torch.randn(long_len, 512)
        batch_lengths = torch.tensor([short_len, long_len], dtype=torch.int64)
        batched = model(
            padded_batch, torch.zeros(2, dtype=torch.int64), input_lengths=batch_lengths,
        )

    alone_valid = alone[:, 0, :]
    batched_valid = batched[:short_len, 0, :]
    return (alone_valid - batched_valid).abs().max().item()


def test_smoothing_batch_composition_sensitivity_below_threshold_after_fix(capsys) -> None:
    """THE FIX, verified: forward() now masks every padded timestep to
    zero immediately after DayLayers and before _smooth runs, so a
    TRAINED (nonzero-bias) DayLayers can no longer leak its bias into the
    padded region for smoothing to mix into real timesteps. Both the
    fresh (identity-init) and trained (nonzero-bias) cases must now be
    below 1e-5 -- reports the measured values (previously 0.0755 for the
    trained case, before this fix; see docs/DESIGN.md's "Model
    hyperparameters" section)."""
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

    print("\n[Task 1] max abs log_probs difference, trial alone vs. "
          "padded in a larger batch, AFTER the padding-mask-before-"
          "smoothing fix:")
    print(f"[Task 1]   fresh (identity-init) DayLayers:  {fresh_diff:.8f}")
    print(f"[Task 1]   trained (nonzero-bias) DayLayers: {trained_diff:.8f}")

    captured = capsys.readouterr()
    assert "max abs log_probs difference" in captured.out
    assert math.isfinite(fresh_diff)
    assert math.isfinite(trained_diff)
    assert fresh_diff < 1e-5
    assert trained_diff < 1e-5


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


# --- day_layer_parameterization plumbing ----------------------------------


@pytest.mark.parametrize("parameterization", ["full", "diagonal", "lowrank"])
def test_day_layer_parameterization_is_forwarded_to_day_layers(parameterization) -> None:
    """GRUCTC must pass day_layer_parameterization through to DayLayers so
    a config can select "diagonal"/"lowrank" (docs/FINDINGS.md: "Day
    layers are unstable when trained jointly across many sessions")."""
    model = GRUCTC(
        n_sessions=3, n_layers=1, hidden_size=16, use_day_layers=True,
        day_layer_parameterization=parameterization, day_layer_rank=4,
    )
    assert model.day_layers.parameterization == parameterization
    # "diagonal" owns a per-channel scale vector, not a square weight matrix.
    if parameterization == "diagonal":
        assert hasattr(model.day_layers, "scale")
        assert not hasattr(model.day_layers, "weights")
    elif parameterization == "full":
        assert model.day_layers.weights.shape == (3, 512, 512)


def test_day_layer_parameterization_default_is_full() -> None:
    model = GRUCTC(n_sessions=2, n_layers=1, hidden_size=16, use_day_layers=True)
    assert model.day_layers.parameterization == "full"


def test_invalid_day_layer_parameterization_raises() -> None:
    with pytest.raises(ValueError, match="parameterization"):
        GRUCTC(n_sessions=1, use_day_layers=True, day_layer_parameterization="bogus")
