"""Unit tests for `b2t_drift.training.checkpoint`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from b2t_drift.training.checkpoint import load_checkpoint, save_checkpoint


def _tiny_model_and_optimizer():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def test_save_then_load_round_trips_model_and_metadata(tmp_path) -> None:
    model, optimizer = _tiny_model_and_optimizer()
    with torch.no_grad():
        model.weight.fill_(0.5)

    metrics = {"val_per": 0.31, "train_loss": 1.2}
    config = {"model": {"hidden_size": 512}, "seed": 0}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, metrics=metrics, config=config)

    reloaded_model, reloaded_optimizer = _tiny_model_and_optimizer()
    result = load_checkpoint(path, reloaded_model, optimizer=reloaded_optimizer)

    assert result["epoch"] == 3
    assert result["metrics"] == metrics
    assert result["config"] == config
    assert torch.equal(reloaded_model.weight, model.weight)


def test_load_without_optimizer_does_not_require_optimizer_state(tmp_path) -> None:
    model, optimizer = _tiny_model_and_optimizer()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, metrics={}, config={})

    reloaded_model, _ = _tiny_model_and_optimizer()
    result = load_checkpoint(path, reloaded_model, optimizer=None)
    assert result["epoch"] == 1


def test_load_raises_if_optimizer_given_but_checkpoint_has_none(tmp_path) -> None:
    model, _ = _tiny_model_and_optimizer()
    path = tmp_path / "checkpoint.pt"
    # Hand-crafted checkpoint missing "optimizer_state_dict", simulating
    # a malformed or foreign checkpoint file.
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 0, "metrics": {}, "config": {}},
        path,
    )

    reloaded_model, reloaded_optimizer = _tiny_model_and_optimizer()
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        load_checkpoint(path, reloaded_model, optimizer=reloaded_optimizer)


def test_creates_parent_directories(tmp_path) -> None:
    model, optimizer = _tiny_model_and_optimizer()
    path = tmp_path / "nested" / "run_dir" / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=0, metrics={}, config={})
    assert path.exists()


def test_weights_only_numpy_core_failure_falls_back_explicitly_with_warning(tmp_path) -> None:
    """Simulates the documented numpy core/_core weights_only=True failure
    (rather than depending on a specific numpy/torch version combination
    to reproduce it for real) to verify the fallback logic itself: a
    matching failure retries with weights_only=False and warns."""
    model, optimizer = _tiny_model_and_optimizer()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=0, metrics={"x": 1}, config={"y": 2})

    real_torch_load = torch.load
    call_count = {"n": 0}

    def fake_torch_load(*args, **kwargs):
        call_count["n"] += 1
        if kwargs.get("weights_only") is True:
            raise RuntimeError("Unsupported global: numpy._core.multiarray.scalar")
        return real_torch_load(*args, **kwargs)

    reloaded_model, reloaded_optimizer = _tiny_model_and_optimizer()
    with patch("torch.load", side_effect=fake_torch_load), pytest.warns(UserWarning, match="numpy"):
        result = load_checkpoint(path, reloaded_model, optimizer=reloaded_optimizer)

    assert call_count["n"] == 2  # first (weights_only=True) failed, retried
    assert result["metrics"] == {"x": 1}


def test_unrelated_load_failure_is_not_swallowed(tmp_path) -> None:
    """A torch.load failure NOT matching the numpy core/_core signature
    must propagate, not be silently retried/hidden."""
    model, optimizer = _tiny_model_and_optimizer()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=0, metrics={}, config={})

    def fake_torch_load(*args, **kwargs):
        raise RuntimeError("completely unrelated disk error")

    reloaded_model, _ = _tiny_model_and_optimizer()
    with patch("torch.load", side_effect=fake_torch_load), pytest.raises(RuntimeError, match="completely unrelated"):
        load_checkpoint(path, reloaded_model)
