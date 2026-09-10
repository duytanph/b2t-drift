"""Tests for `b2t_drift.training.loop`, against real T15 data (small
subsets) for the end-to-end behaviors, and hand-crafted synthetic batches
for the specific failure modes (infinite loss, sampler.set_epoch) that are
easier to force deterministically than to find in real data.

Real-data tests are skipped if `data/raw/hdf5_data_final` is not present.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
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


# --- train_one_epoch: day_layer_penalty -------------------------------------


def test_day_layer_penalty_zero_leaves_loss_unaffected() -> None:
    """day_layer_penalty=0.0 (the default) must be byte-for-byte the same
    training objective as before this feature existed -- verified by
    comparing loss trajectories against day_layer_penalty simply absent."""
    _require_real_data()
    dataloader, _ = _build_dataloader(n_trials=2, batch_size=2)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    torch.manual_seed(0)
    model_a = _tiny_model()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    result_a = train_one_epoch(model_a, dataloader, optimizer_a, criterion, torch.device("cpu"))

    torch.manual_seed(0)
    model_b = _tiny_model()
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    result_b = train_one_epoch(
        model_b, dataloader, optimizer_b, criterion, torch.device("cpu"), day_layer_penalty=0.0,
    )

    assert result_a["loss"] == pytest.approx(result_b["loss"])


def test_day_layer_penalty_positive_changes_loss_and_pulls_toward_identity() -> None:
    """A nonzero day_layer_penalty must (a) change the reported train loss
    relative to an identical run without it, since the penalty is added
    BEFORE backward, and (b) actually reduce max|W-I| after a training
    step that would otherwise have moved weights away from identity --
    the whole point of the regularizer.

    Both models start PRE-PERTURBED away from identity (not fresh) --
    a freshly-initialized DayLayers sits exactly AT identity, so
    identity_penalty() is exactly 0.0 at step 0 regardless of lambda,
    which would make this comparison vacuous. A SINGLE step is not
    enough to see the pull either: the perturbation here is uniform
    across all 512*512 entries, and one Adam step barely dents it
    relative to that scale (verified directly -- both conditions land
    at the same max|W-I| to 6 decimal places after just one step), so
    this trains for several epochs on the same tiny 2-trial batch to
    let the effect accumulate to something unambiguous.
    """
    _require_real_data()
    dataloader, _ = _build_dataloader(n_trials=2, batch_size=2)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    n_epochs = 20

    def _perturbed_model() -> GRUCTC:
        torch.manual_seed(0)
        model = _tiny_model()
        with torch.no_grad():
            model.day_layers.weights[0] += 0.05
        return model

    model_plain = _perturbed_model()
    optimizer_plain = torch.optim.Adam(model_plain.parameters(), lr=1e-2)
    first_loss_plain = None
    for epoch in range(n_epochs):
        result = train_one_epoch(
            model_plain, dataloader, optimizer_plain, criterion, torch.device("cpu"), epoch=epoch,
        )
        first_loss_plain = first_loss_plain if first_loss_plain is not None else result["loss"]

    model_penalized = _perturbed_model()
    optimizer_penalized = torch.optim.Adam(model_penalized.parameters(), lr=1e-2)
    first_loss_penalized = None
    for epoch in range(n_epochs):
        result = train_one_epoch(
            model_penalized, dataloader, optimizer_penalized, criterion, torch.device("cpu"),
            epoch=epoch, day_layer_penalty=50.0,
        )
        first_loss_penalized = first_loss_penalized if first_loss_penalized is not None else result["loss"]

    assert first_loss_plain != pytest.approx(first_loss_penalized)
    assert first_loss_penalized > first_loss_plain, (
        "the penalty term is strictly additive and positive (a squared "
        "deviation), so the penalized run's first-epoch loss must be higher"
    )

    _, plain_max_dev = model_plain.day_layers.weight_deviation_stats()
    _, penalized_max_dev = model_penalized.day_layers.weight_deviation_stats()
    assert penalized_max_dev < plain_max_dev, (
        f"day_layer_penalty=50.0 over {n_epochs} epochs should pull weights "
        f"closer to identity than no penalty: penalized max|W-I|="
        f"{penalized_max_dev:.6f} was not < plain max|W-I|={plain_max_dev:.6f}"
    )


def test_day_layer_penalty_noop_when_day_layers_disabled() -> None:
    """day_layer_penalty > 0 with use_day_layers=False must not raise --
    there is nothing to regularize, so it's silently a no-op, matching
    the docstring's stated contract."""
    _require_real_data()
    dataloader, _ = _build_dataloader(n_trials=2, batch_size=2)
    model = GRUCTC(n_sessions=1, hidden_size=16, n_layers=1, dropout=0.0, use_day_layers=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    result = train_one_epoch(
        model, dataloader, optimizer, criterion, torch.device("cpu"), day_layer_penalty=10.0,
    )
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

    assert set(result) == {
        "loss", "per", "counts", "counts_by_session", "n_trials", "seconds",
        "blank_fraction", "decoded_to_reference_length_ratio", "decoded_token_counts",
        "silence_deletions", "phoneme_deletions",
    }
    assert result["n_trials"] == 2
    assert len(result["counts"]) == 2
    assert SESSION_ID in result["counts_by_session"]
    assert len(result["counts_by_session"][SESSION_ID]) == 2
    assert result["per"] < 0.2, f"expected a well-overfit model to have low PER, got {result['per']}"
    # A well-overfit model should have moved well off the all-blank
    # starting point and decode roughly reference-length output.
    assert 0.0 <= result["blank_fraction"] < 1.0
    assert result["decoded_to_reference_length_ratio"] > 0.0
    assert sum(result["decoded_token_counts"].values()) > 0
    total_deletions = sum(c.deletions for c in result["counts"])
    assert result["silence_deletions"] + result["phoneme_deletions"] == total_deletions


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
    assert result["blank_fraction"] is None
    assert result["decoded_to_reference_length_ratio"] is None
    assert result["decoded_token_counts"] == {}
    assert result["silence_deletions"] is None
    assert result["phoneme_deletions"] is None


class _FixedLogProbsModel(nn.Module):
    """Returns a precomputed `log_probs` tensor regardless of input.

    Lets `evaluate`'s frame-level diagnostics (blank_fraction and friends)
    be tested against an exact, hand-computed `log_probs` tensor, instead
    of a real GRU's opaque output where the desired argmax pattern can't
    be dictated directly.
    """

    def __init__(self, log_probs: torch.Tensor, n_classes: int) -> None:
        super().__init__()
        self._log_probs = log_probs
        self.n_classes = n_classes

    def ctc_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return input_lengths

    def forward(self, features, session_indices, input_lengths=None):
        return self._log_probs


def test_evaluate_all_blank_decode_gives_per_exactly_one() -> None:
    """A model forced to argmax blank at every valid timestep: blank_fraction
    == 1.0, the decode is empty (all deletions), and PER == 1.0 exactly --
    pinning down the reference point every other blank_fraction reading is
    interpreted against."""
    n_classes = 41
    logits = torch.full((5, 1, n_classes), -20.0)
    logits[:, :, BLANK_INDEX] = 20.0  # argmax == blank at every timestep
    log_probs = F.log_softmax(logits, dim=-1)

    batch = {
        "features": torch.zeros(1, 5, 512),
        "targets": torch.tensor([1, 2, 3], dtype=torch.int64),
        "input_lengths": torch.tensor([5], dtype=torch.int64),
        "target_lengths": torch.tensor([3], dtype=torch.int64),
        "session_indices": torch.zeros(1, dtype=torch.int64),
        "session_ids": ["fake"],
        "block_nums": [1],
        "trial_keys": ["trial_0000"],
    }
    model = _FixedLogProbsModel(log_probs, n_classes=n_classes)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    result = evaluate(model, [batch], criterion, torch.device("cpu"), compute_per=True)

    assert result["blank_fraction"] == 1.0
    assert result["decoded_to_reference_length_ratio"] == 0.0
    assert result["decoded_token_counts"] == {}
    assert result["per"] == 1.0  # empty decode against 3 reference phonemes: 3 deletions / 3 == 1.0
    # targets [1, 2, 3] contain no silence (class 40) -- all 3 deletions are real phonemes.
    assert result["silence_deletions"] == 0
    assert result["phoneme_deletions"] == 3


def test_evaluate_blank_fraction_ignores_padded_timesteps() -> None:
    """Two items of very different lengths, where the SHORTER item's padded
    region is deliberately given the OPPOSITE argmax pattern from its valid
    region -- if evaluate() failed to mask padding (e.g. used the wrong
    length tensor, or none at all), blank_fraction would come out
    measurably different from the hand-computed valid-timesteps-only value.
    """
    n_classes = 41
    time, batch_size = 6, 2
    logits = torch.full((time, batch_size, n_classes), -20.0)

    # Item 0: input_length=3. Valid region (t=0,1,2): all non-blank (class
    # 5). Padded region (t=3,4,5): all BLANK -- included only if masking is
    # broken.
    logits[0, 0, 5] = 20.0
    logits[1, 0, 5] = 20.0
    logits[2, 0, 5] = 20.0
    logits[3, 0, BLANK_INDEX] = 20.0
    logits[4, 0, BLANK_INDEX] = 20.0
    logits[5, 0, BLANK_INDEX] = 20.0

    # Item 1: input_length=6 (no padding). 3 blank frames, then 3 non-blank
    # (class 7).
    logits[0, 1, BLANK_INDEX] = 20.0
    logits[1, 1, BLANK_INDEX] = 20.0
    logits[2, 1, BLANK_INDEX] = 20.0
    logits[3, 1, 7] = 20.0
    logits[4, 1, 7] = 20.0
    logits[5, 1, 7] = 20.0

    log_probs = F.log_softmax(logits, dim=-1)

    batch = {
        "features": torch.zeros(2, 6, 512),
        "targets": torch.tensor([5, 7], dtype=torch.int64),
        "input_lengths": torch.tensor([3, 6], dtype=torch.int64),
        "target_lengths": torch.tensor([1, 1], dtype=torch.int64),
        "session_indices": torch.zeros(2, dtype=torch.int64),
        "session_ids": ["fake0", "fake1"],
        "block_nums": [1, 1],
        "trial_keys": ["trial_0000", "trial_0001"],
    }
    model = _FixedLogProbsModel(log_probs, n_classes=n_classes)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    result = evaluate(model, [batch], criterion, torch.device("cpu"), compute_per=True)

    # Hand-computed, valid-timesteps-only: item 0 contributes 3 valid
    # frames, 0 blank; item 1 contributes 6 valid frames, 3 blank.
    # 3 blank / 9 valid == 1/3. Counting item 0's padded region (the bug
    # this test catches) would instead give 6 blank / 12 == 0.5.
    expected = 3 / 9
    assert result["blank_fraction"] == pytest.approx(expected)
    assert result["blank_fraction"] != pytest.approx(0.5)


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
        "lr", "seconds", "train_seconds", "eval_seconds", "n_inf_batches",
        "val_blank_fraction", "val_decoded_length_ratio", "val_n_distinct_decoded_tokens",
        "day_penalty", "day_layer_mean_max_weight_dev", "day_layer_max_weight_dev",
    }
    assert set(rows[0].keys()) == expected_columns
    # _tiny_model() has use_day_layers=True -- these must be real numbers,
    # not empty, whether or not day_layer_penalty is actually applied.
    assert rows[0]["day_penalty"] not in ("", None)
    assert rows[0]["day_layer_mean_max_weight_dev"] not in ("", None)

    counts_path = tmp_path / "decoded_token_counts.jsonl"
    assert counts_path.exists()
    with counts_path.open() as f:
        jsonl_rows = [json.loads(line) for line in f]
    assert len(jsonl_rows) == 3
    assert [r["epoch"] for r in jsonl_rows] == [0, 1, 2]
    assert all(isinstance(r["counts"], dict) for r in jsonl_rows)


def test_fit_day_layer_columns_are_none_when_day_layers_disabled(tmp_path) -> None:
    _require_real_data()
    torch.manual_seed(0)
    train_loader, sampler = _build_dataloader(n_trials=4, batch_size=2, bucketed=True)
    val_loader, _ = _build_dataloader(n_trials=2, batch_size=2)
    model = GRUCTC(n_sessions=1, hidden_size=16, n_layers=1, dropout=0.0, use_day_layers=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    fit(
        model, train_loader, val_loader, optimizer, criterion, torch.device("cpu"),
        max_epochs=1, patience=10, run_dir=tmp_path, train_sampler=sampler, config={"seed": 0},
    )

    with (tmp_path / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["day_penalty"] == ""
    assert rows[0]["day_layer_mean_max_weight_dev"] == ""
    assert rows[0]["day_layer_max_weight_dev"] == ""


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
        return {
            "loss": 1.0, "per": per, "counts": [], "counts_by_session": {}, "n_trials": 0, "seconds": 0.0,
            "blank_fraction": 0.5, "decoded_to_reference_length_ratio": 1.0, "decoded_token_counts": {},
        }

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
