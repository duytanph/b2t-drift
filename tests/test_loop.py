"""Tests for `b2t_drift.training.loop`, against real T15 data (small
subsets) for the end-to-end behaviors, and hand-crafted synthetic batches
for the specific failure modes (infinite loss, sampler.set_epoch) that are
easier to force deterministically than to find in real data.

Real-data tests are skipped if `data/raw/hdf5_data_final` is not present.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from b2t_drift.data.collate import LengthBucketedSampler, collate_trials
from b2t_drift.data.dataset import TrialDataset
from b2t_drift.data.hdf5_io import iter_trial_ids
from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.phonemes import BLANK_INDEX
from b2t_drift.training.loop import evaluate, fit, train_one_epoch

DATA_ROOT = Path("data/raw/hdf5_data_final")
SESSION_ID = "t15.2023.08.11"


def _require_real_data() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")


def _build_dataloader(n_trials: int, batch_size: int = 2, bucketed: bool = False):
    trial_ids = list(iter_trial_ids(DATA_ROOT, SESSION_ID))[:n_trials]
    dataset = TrialDataset(DATA_ROOT, trial_ids, {SESSION_ID: 0})
    if bucketed:
        sampler = LengthBucketedSampler(trial_ids, batch_size=batch_size, drop_last=False)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_trials), sampler
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_trials), None


def _tiny_model() -> GRUCTC:
    return GRUCTC(n_sessions=1, hidden_size=16, n_layers=1, dropout=0.0, use_day_layers=True)


# --- train_one_epoch ---------------------------------------------------


def test_train_one_epoch_returns_expected_keys_and_reduces_loss() -> None:
    _require_real_data()
    torch.manual_seed(0)
    dataloader, _ = _build_dataloader(n_trials=4, batch_size=2)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    device = torch.device("cpu")

    losses = []
    for epoch in range(5):
        result = train_one_epoch(model, dataloader, optimizer, criterion, device, epoch=epoch)
        assert set(result) == {"loss", "n_batches", "n_inf_batches", "grad_norm_mean", "seconds"}
        assert result["n_batches"] == 2
        assert result["n_inf_batches"] == 0
        assert result["grad_norm_mean"] >= 0
        losses.append(result["loss"])

    assert losses[-1] < losses[0]


def test_train_one_epoch_calls_sampler_set_epoch() -> None:
    _require_real_data()
    dataloader, sampler = _build_dataloader(n_trials=4, batch_size=2, bucketed=True)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    calls: list[int] = []
    original_set_epoch = sampler.set_epoch

    def spy_set_epoch(epoch: int) -> None:
        calls.append(epoch)
        original_set_epoch(epoch)

    sampler.set_epoch = spy_set_epoch

    train_one_epoch(
        model, dataloader, optimizer, criterion, torch.device("cpu"), epoch=3, sampler=sampler,
    )
    assert calls == [3]


def test_train_one_epoch_warns_on_zero_infinity_false() -> None:
    _require_real_data()
    dataloader, _ = _build_dataloader(n_trials=2, batch_size=2)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=False)

    with pytest.warns(UserWarning, match="zero_infinity"):
        train_one_epoch(model, dataloader, optimizer, criterion, torch.device("cpu"))


def test_train_one_epoch_counts_and_skips_inf_batches() -> None:
    """Hand-crafted batch with target_length > input_length (CTC's hard
    requirement violated) forces a genuinely infinite loss -- verifies
    it's counted and skipped rather than crashing the epoch or corrupting
    the model."""
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    bad_batch = {
        "features": torch.randn(1, 3, 512),  # only 3 timesteps
        "targets": torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),  # 5 target tokens
        "input_lengths": torch.tensor([3], dtype=torch.int64),
        "target_lengths": torch.tensor([5], dtype=torch.int64),
        "session_indices": torch.zeros(1, dtype=torch.int64),
        "session_ids": ["fake"],
        "block_nums": [1],
        "trial_keys": ["trial_0000"],
    }
    before = {name: p.clone() for name, p in model.named_parameters()}

    result = train_one_epoch(model, [bad_batch], optimizer, criterion, torch.device("cpu"))

    assert result["n_batches"] == 1
    assert result["n_inf_batches"] == 1
    assert math.isnan(result["loss"])  # every batch was inf -- see the docstring
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]), f"{name} changed despite the batch being skipped"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_train_one_epoch_amp_on_cuda_runs_and_produces_finite_loss() -> None:
    _require_real_data()
    torch.manual_seed(0)
    dataloader, _ = _build_dataloader(n_trials=4, batch_size=2)
    device = torch.device("cuda")
    model = _tiny_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda")

    result = train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=scaler)
    assert result["n_inf_batches"] == 0
    assert math.isfinite(result["loss"])


# --- evaluate ------------------------------------------------------------


def test_evaluate_returns_expected_keys_and_low_per_after_overfitting() -> None:
    _require_real_data()
    torch.manual_seed(0)
    dataloader, _ = _build_dataloader(n_trials=2, batch_size=2)
    model = GRUCTC(n_sessions=1, hidden_size=64, n_layers=1, dropout=0.0, use_day_layers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    device = torch.device("cpu")

    for _ in range(150):
        train_one_epoch(model, dataloader, optimizer, criterion, device)

    result = evaluate(model, dataloader, criterion, device, blank_index=BLANK_INDEX, compute_per=True)

    assert set(result) == {"loss", "per", "counts", "counts_by_session", "n_trials", "seconds"}
    assert result["n_trials"] == 2
    assert len(result["counts"]) == 2
    assert SESSION_ID in result["counts_by_session"]
    assert len(result["counts_by_session"][SESSION_ID]) == 2
    assert result["per"] < 0.2, f"expected a well-overfit model to have low PER, got {result['per']}"


def test_evaluate_compute_per_false_skips_decode() -> None:
    _require_real_data()
    dataloader, _ = _build_dataloader(n_trials=2, batch_size=2)
    model = _tiny_model()
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    result = evaluate(model, dataloader, criterion, torch.device("cpu"), compute_per=False)

    assert result["per"] is None
    assert result["counts"] == []
    assert result["counts_by_session"] == {}
    assert result["n_trials"] == 2


# --- fit -------------------------------------------------------------------


def test_fit_smoke_run_creates_checkpoints_and_metrics_csv(tmp_path) -> None:
    _require_real_data()
    torch.manual_seed(0)
    train_loader, sampler = _build_dataloader(n_trials=4, batch_size=2, bucketed=True)
    val_loader, _ = _build_dataloader(n_trials=2, batch_size=2)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    summary = fit(
        model, train_loader, val_loader, optimizer, criterion, torch.device("cpu"),
        max_epochs=3, patience=10, run_dir=tmp_path, train_sampler=sampler,
        config={"seed": 0},
    )

    assert set(summary) == {"best_epoch", "best_val_per", "final_epoch", "total_seconds"}
    assert summary["best_epoch"] is not None
    assert summary["final_epoch"] == 2  # ran all 3 epochs, patience=10 never triggers

    assert (tmp_path / "checkpoint_best.pt").exists()
    assert (tmp_path / "checkpoint_last.pt").exists()

    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.exists()
    with metrics_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    expected_columns = {
        "epoch", "train_loss", "val_loss", "val_per", "grad_norm_mean",
        "lr", "seconds", "n_inf_batches",
    }
    assert set(rows[0].keys()) == expected_columns


def test_fit_stops_early_when_val_per_does_not_improve(tmp_path, monkeypatch) -> None:
    """Mocks train_one_epoch/evaluate with a canned val_per sequence to
    deterministically exercise the early-stopping boundary, rather than
    hoping real data happens to plateau at the right moment."""
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    # epoch 0: 0.9 -> improves (baseline). epoch 1: 0.5 -> improves.
    # epoch 2: 0.5 -> no improvement (count=1). epoch 3: 0.5 -> no
    # improvement (count=2) -> STOPS (patience=2). epoch 4 never runs.
    per_sequence = [0.9, 0.5, 0.5, 0.5, 0.1]
    calls = {"n": 0}

    def fake_evaluate(*args, **kwargs):
        per = per_sequence[calls["n"]]
        calls["n"] += 1
        return {"loss": 1.0, "per": per, "counts": [], "counts_by_session": {}, "n_trials": 0, "seconds": 0.0}

    def fake_train_one_epoch(*args, **kwargs):
        return {"loss": 1.0, "n_batches": 1, "n_inf_batches": 0, "grad_norm_mean": 0.1, "seconds": 0.0}

    monkeypatch.setattr("b2t_drift.training.loop.evaluate", fake_evaluate)
    monkeypatch.setattr("b2t_drift.training.loop.train_one_epoch", fake_train_one_epoch)

    summary = fit(
        model, [], [], optimizer, criterion, torch.device("cpu"),
        max_epochs=10, patience=2, run_dir=tmp_path,
    )

    assert calls["n"] == 4  # stopped after epoch 3, epoch 4 never evaluated
    assert summary["final_epoch"] == 3
    assert summary["best_epoch"] == 1
    assert summary["best_val_per"] == 0.5
