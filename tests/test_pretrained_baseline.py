"""Unit tests for `b2t_drift.models.pretrained_baseline`.

Uses a small FABRICATED checkpoint (correct key names/shapes, `n_sessions=2`
instead of the real 45, real feature/hidden/layer dimensions since those are
fixed constants on `PretrainedBaselineRNN`) rather than the real ~500 MB
released file, so these tests run fast and do not depend on
`data/external/` being present. The real checkpoint is exercised by
`scratch/verify_pretrained_checkpoint.py` (not committed -- gitignored,
`CLAUDE.md`), whose result (adapter accounts for all 113 real parameters;
`transpose_day_weights=False` reproduces the checkpoint's recorded
`val_PER=0.101` on held-out sessions) is reported separately.
"""

from __future__ import annotations

import pytest
import torch

from b2t_drift.models.pretrained_baseline import (
    HIDDEN_SIZE,
    N_CLASSES,
    N_FEATURES,
    N_LAYERS,
    PATCH_SIZE,
    PATCH_STRIDE,
    PretrainedBaselineRNN,
    build_pretrained_baseline_model,
    load_pretrained_baseline_checkpoint,
)

N_SESSIONS = 2


def _fake_raw_state_dict(n_sessions: int = N_SESSIONS, seed: int = 0) -> dict[str, torch.Tensor]:
    """Build a fabricated `_orig_mod.`-prefixed state dict matching the
    released checkpoint's key names and shapes exactly (see
    `scratch/inspect_checkpoint.py`'s output), with `n_sessions` day-layer
    slices instead of the real 45.
    """
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    sd["h0"] = torch.randn(1, 1, HIDDEN_SIZE, generator=g)
    for i in range(n_sessions):
        # Deliberately NOT identity/symmetric, so a transpose is detectable.
        sd[f"day_weights.{i}"] = torch.randn(N_FEATURES, N_FEATURES, generator=g)
        sd[f"day_biases.{i}"] = torch.randn(1, N_FEATURES, generator=g)
    gru_input_size = N_FEATURES * PATCH_SIZE
    for layer in range(N_LAYERS):
        in_size = gru_input_size if layer == 0 else HIDDEN_SIZE
        sd[f"gru.weight_ih_l{layer}"] = torch.randn(3 * HIDDEN_SIZE, in_size, generator=g)
        sd[f"gru.weight_hh_l{layer}"] = torch.randn(3 * HIDDEN_SIZE, HIDDEN_SIZE, generator=g)
        sd[f"gru.bias_ih_l{layer}"] = torch.randn(3 * HIDDEN_SIZE, generator=g)
        sd[f"gru.bias_hh_l{layer}"] = torch.randn(3 * HIDDEN_SIZE, generator=g)
    sd["out.weight"] = torch.randn(N_CLASSES, HIDDEN_SIZE, generator=g)
    sd["out.bias"] = torch.randn(N_CLASSES, generator=g)
    return {f"_orig_mod.{k}": v for k, v in sd.items()}


def _write_fake_checkpoint(tmp_path, raw_sd) -> "type(tmp_path)":
    path = tmp_path / "fake_checkpoint"
    torch.save(
        {
            "model_state_dict": raw_sd,
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "val_PER": 0.1234,
            "val_loss": 5.678,
        },
        path,
    )
    return path


def test_build_pretrained_baseline_model_matches_checkpoint_architecture() -> None:
    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    assert isinstance(model, PretrainedBaselineRNN)
    assert model.hidden_size == HIDDEN_SIZE
    assert model.n_layers == N_LAYERS
    assert model.patch_size == PATCH_SIZE
    assert model.patch_stride == PATCH_STRIDE
    assert model.n_classes == N_CLASSES
    assert model.day_layer_parameterization == "full"
    assert model.h0.shape == (1, 1, HIDDEN_SIZE)


def test_adapter_maps_every_parameter_and_values_round_trip(tmp_path) -> None:
    raw_sd = _fake_raw_state_dict()
    path = _write_fake_checkpoint(tmp_path, raw_sd)

    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    info = load_pretrained_baseline_checkpoint(path, model, transpose_day_weights=False)

    assert info["n_checkpoint_params"] == len(raw_sd) == info["n_mapped_params"]
    assert info["val_PER"] == pytest.approx(0.1234)
    assert info["val_loss"] == pytest.approx(5.678)

    # h0: direct copy.
    assert torch.equal(model.h0.data, raw_sd["_orig_mod.h0"])

    # Day layers: stacked, un-transposed (this test's default).
    for i in range(N_SESSIONS):
        assert torch.equal(model.day_layers.weights.data[i], raw_sd[f"_orig_mod.day_weights.{i}"])
        assert torch.equal(model.day_layers.biases.data[i], raw_sd[f"_orig_mod.day_biases.{i}"].squeeze(0))

    # GRU: renamed prefix only, values unchanged.
    for layer in range(N_LAYERS):
        for part in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            key = f"gru.{part}_l{layer}"
            actual = dict(model.gru.named_parameters())[f"{part}_l{layer}"]
            assert torch.equal(actual.data, raw_sd[f"_orig_mod.{key}"])

    # Output layer: "out" -> "output" rename.
    assert torch.equal(model.output.weight.data, raw_sd["_orig_mod.out.weight"])
    assert torch.equal(model.output.bias.data, raw_sd["_orig_mod.out.bias"])


def test_adapter_transpose_day_weights_flag_actually_transposes(tmp_path) -> None:
    raw_sd = _fake_raw_state_dict()
    path = _write_fake_checkpoint(tmp_path, raw_sd)

    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    load_pretrained_baseline_checkpoint(path, model, transpose_day_weights=True)

    for i in range(N_SESSIONS):
        expected = raw_sd[f"_orig_mod.day_weights.{i}"].T
        assert torch.equal(model.day_layers.weights.data[i], expected)


def test_adapter_raises_on_missing_key_naming_it(tmp_path) -> None:
    raw_sd = _fake_raw_state_dict()
    del raw_sd["_orig_mod.gru.weight_hh_l2"]
    path = _write_fake_checkpoint(tmp_path, raw_sd)

    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    with pytest.raises(ValueError, match="gru.weight_hh_l2"):
        load_pretrained_baseline_checkpoint(path, model)


def test_adapter_raises_on_missing_day_layer_key_naming_session_index(tmp_path) -> None:
    raw_sd = _fake_raw_state_dict()
    del raw_sd["_orig_mod.day_biases.1"]
    path = _write_fake_checkpoint(tmp_path, raw_sd)

    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    with pytest.raises(ValueError, match="session index 1"):
        load_pretrained_baseline_checkpoint(path, model)


def test_adapter_raises_on_unmapped_extra_key_naming_it(tmp_path) -> None:
    raw_sd = _fake_raw_state_dict()
    raw_sd["_orig_mod.some_new_module.weight"] = torch.randn(3, 3)
    path = _write_fake_checkpoint(tmp_path, raw_sd)

    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    with pytest.raises(ValueError, match="some_new_module.weight"):
        load_pretrained_baseline_checkpoint(path, model)


def test_adapter_raises_on_missing_prefix(tmp_path) -> None:
    raw_sd = _fake_raw_state_dict()
    raw_sd["no_prefix_here.weight"] = raw_sd.pop("_orig_mod.h0")
    path = _write_fake_checkpoint(tmp_path, raw_sd)

    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    with pytest.raises(ValueError, match="_orig_mod"):
        load_pretrained_baseline_checkpoint(path, model)


def test_forward_runs_end_to_end_with_loaded_weights(tmp_path) -> None:
    """Loads a fake checkpoint and runs a real forward pass, checking
    output shape and that the learned h0 is actually used (see
    `PretrainedBaselineRNN._run_gru`) -- a nonzero h0 must change the
    output relative to a zeroed one, for at least a short sequence where
    the initial state hasn't been washed out by the recurrence.
    """
    raw_sd = _fake_raw_state_dict()
    path = _write_fake_checkpoint(tmp_path, raw_sd)

    model = build_pretrained_baseline_model(n_sessions=N_SESSIONS)
    load_pretrained_baseline_checkpoint(path, model)
    model.eval()

    batch, time = 3, 40  # >= patch_size=14 so at least one patch exists
    features = torch.randn(batch, time, N_FEATURES)
    session_indices = torch.tensor([0, 1, 0], dtype=torch.int64)
    input_lengths = torch.tensor([time, time - 5, time - 10], dtype=torch.int64)

    with torch.no_grad():
        log_probs = model(features, session_indices, input_lengths=input_lengths)
    expected_time = (time - PATCH_SIZE) // PATCH_STRIDE + 1
    assert log_probs.shape == (expected_time, batch, N_CLASSES)
    assert torch.isfinite(log_probs).all()

    with torch.no_grad():
        log_probs_zero_h0 = model(features, session_indices, input_lengths=input_lengths)
        model.h0.zero_()
        log_probs_after_zeroing = model(features, session_indices, input_lengths=input_lengths)
    assert not torch.equal(log_probs_zero_h0, log_probs_after_zeroing), (
        "zeroing h0 should change output for a model whose loaded h0 was nonzero"
    )
