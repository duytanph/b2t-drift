"""THE PIPELINE GATE: the model must be able to overfit two real trials.

RATIONALE FOR THE TEST ITSELF. If a network cannot memorize two examples,
something upstream is broken -- transposed dimensions, swapped input and
target lengths, an off-by-one phoneme mapping, or logits on the wrong
axis. All of these produce a model that trains, converges, and emits
garbage WITHOUT raising an error (`CLAUDE.md`'s "known pitfalls";
`data/collate.py`'s module docstring makes the same point about the
collation step this test also exercises). This is the pipeline gate: run
it before every real training job, not just once at implementation time.

THRESHOLD RATIONALE:
  - 300 steps: ample for a small model (`n_layers=2, hidden_size=256`) to
    memorize two examples; needing more would itself indicate a problem.
  - final loss < 0.5: CTC loss is approximately negative log-likelihood
    per sequence; a memorized pair should drive it near zero, and 0.5 is
    loose enough to avoid flakiness while still failing clearly on a
    broken pipeline.
  - final loss < 10% of initial loss: catches the case where the loss is
    already small at initialization (e.g. a degenerate model that always
    predicts blank against a short target) without ever having learned
    anything -- the absolute threshold alone wouldn't catch that.
  - lr=1e-3 with Adam: standard starting point; arbitrary here since any
    converging optimizer suffices for a 2-example memorization task.

Real-data tests (the gate itself, the 20-step-window check, the decode
check, and the swapped-lengths control) are skipped if
`data/raw/hdf5_data_final` is not present; unit tests of `DayLayers` /
`GRUCTC` in isolation (targeting silent failures in the model code itself,
independent of any specific data) use synthetic tensors and always run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from b2t_drift.data.collate import collate_trials
from b2t_drift.data.hdf5_io import iter_trial_ids, read_trial
from b2t_drift.evaluation.ctc_decode import greedy_ctc_decode
from b2t_drift.models.day_layers import DayLayers
from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.phonemes import BLANK_INDEX, N_CLASSES

DATA_ROOT = Path("data/raw/hdf5_data_final")
SESSION_ID = "t15.2023.08.11"  # 288 labeled trials, train partition only

N_STEPS = 300
LOSS_ABSOLUTE_THRESHOLD = 0.5
LOSS_RELATIVE_THRESHOLD = 0.10
LEARNING_RATE = 1e-3


def _require_real_data() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")


def _load_two_items() -> list[dict]:
    """Two real trials from `SESSION_ID`, shaped as `TrialDataset` items.

    `session_index` is assigned 0 and 1 (NOT looked up from a real
    `session_to_index` mapping, which would give both trials the same
    index since they're from the same physical session) so the gate
    genuinely exercises two distinct `DayLayers` slices, as the task
    driving this test (`n_sessions=2`) requires.
    """
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


def _build_gate_model() -> GRUCTC:
    return GRUCTC(
        n_sessions=2,
        hidden_size=256,
        n_layers=2,
        dropout=0.0,
        use_day_layers=True,
    )


@pytest.fixture(scope="module")
def gate_run() -> dict:
    """Train `GRUCTC` for `N_STEPS` on 2 real trials -- the pipeline gate."""
    _require_real_data()
    torch.manual_seed(0)

    items = _load_two_items()
    batch = collate_trials(items)
    model = _build_gate_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=False)

    losses = []
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        log_probs = model(batch["features"], batch["session_indices"])
        loss = ctc_loss_fn(
            log_probs,
            batch["targets"],
            model.ctc_input_lengths(batch["input_lengths"]),
            batch["target_lengths"],
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return {"losses": losses, "model": model, "batch": batch, "items": items}


# --- THE GATE ----------------------------------------------------------


def test_overfit_two_real_trials_gate(gate_run) -> None:
    """THE GATE: final loss must collapse both in absolute and relative terms."""
    losses = gate_run["losses"]
    initial, final = losses[0], losses[-1]
    assert final < LOSS_ABSOLUTE_THRESHOLD, (
        f"final loss {final:.4f} did not drop below the absolute "
        f"threshold {LOSS_ABSOLUTE_THRESHOLD} -- the model failed to "
        f"memorize two real trials; see this module's docstring."
    )
    assert final < LOSS_RELATIVE_THRESHOLD * initial, (
        f"final loss {final:.4f} did not drop below "
        f"{LOSS_RELATIVE_THRESHOLD:.0%} of the initial loss {initial:.4f}."
    )


# --- (1) loss decreases over a window, not just endpoint-to-endpoint ---


def test_loss_decreases_over_a_20_step_window(gate_run) -> None:
    losses = gate_run["losses"]
    assert losses[19] < losses[0], "loss did not decrease over the first 20 steps"
    mid = len(losses) // 2
    assert losses[mid + 19] < losses[mid], "loss did not decrease over a mid-run 20-step window"


# --- (2) greedy decode recovers the correct phoneme sequence -----------


def test_greedy_decode_recovers_at_least_one_trial(gate_run) -> None:
    """The strongest end-to-end check: verifies label SEMANTICS, not just
    that some loss number went down."""
    model = gate_run["model"]
    batch = gate_run["batch"]
    items = gate_run["items"]

    model.eval()
    with torch.no_grad():
        log_probs = model(batch["features"], batch["session_indices"])
    model.train()

    decoded = greedy_ctc_decode(
        log_probs.transpose(0, 1),  # (time, batch, C) -> (batch, time, C)
        blank_index=BLANK_INDEX,
        lengths=batch["input_lengths"],
    )
    true_targets = [item["targets"].tolist() for item in items]

    assert any(d == t for d, t in zip(decoded, true_targets)), (
        f"no trial's greedy decode matched its true phoneme sequence.\n"
        f"decoded: {decoded}\ntrue:    {true_targets}"
    )


# --- (3) swapped-lengths control -----------------------------------------


def test_swapped_lengths_control_does_not_reach_threshold() -> None:
    """Proves the gate would actually catch the bug it exists to catch.

    Same 300-step run as the gate, but with `input_lengths` and
    `target_lengths` swapped when calling `nn.CTCLoss`. Two outcomes are
    both acceptable evidence the gate works, and this test accepts
    either, asserting on whichever was actually observed:
      - `nn.CTCLoss` raises, because the swapped `target_lengths` (the
        real trials' large `n_time_steps` values) no longer sum to the
        concatenated `targets` tensor's actual length (which is still the
        sum of the real, small `seq_len` values) -- an internal
        consistency check catches the corruption immediately.
      - Training proceeds without raising, but the loss never reaches the
        gate's threshold, because the model is now optimizing against a
        length contract that doesn't match its labels.
    Either way, the swap must NOT quietly produce a passing gate.
    """
    _require_real_data()
    torch.manual_seed(0)

    items = _load_two_items()
    batch = collate_trials(items)
    model = _build_gate_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=False)

    swapped_input_lengths = batch["target_lengths"]
    swapped_target_lengths = batch["input_lengths"]

    losses: list[float] = []
    raised: Exception | None = None
    try:
        for _ in range(N_STEPS):
            optimizer.zero_grad()
            log_probs = model(batch["features"], batch["session_indices"])
            loss = ctc_loss_fn(
                log_probs,
                batch["targets"],
                model.ctc_input_lengths(swapped_input_lengths),
                swapped_target_lengths,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    except RuntimeError as exc:
        raised = exc

    if raised is not None:
        # OBSERVED BEHAVIOR (recorded here, not just asserted blindly):
        # nn.CTCLoss raises immediately, on the very first step, because
        # the swapped target_lengths (the real trials' large
        # n_time_steps values) no longer sum to the concatenated targets
        # tensor's actual size (which is still the sum of the real,
        # small seq_len values) -- e.g. "Expected tensor to have size
        # 802 at dimension 0, but got size 33 for argument #2 'targets'".
        # This IS the gate catching the error, just earlier than a full
        # 300-step run: assert the raise is actually about this
        # size/length inconsistency, not some unrelated failure.
        message = str(raised).lower()
        assert "size" in message or "length" in message or "target" in message, (
            f"nn.CTCLoss raised, but not with a size/length-mismatch "
            f"message as expected: {raised}"
        )
    else:
        final = losses[-1]
        gate_would_have_passed = final < LOSS_ABSOLUTE_THRESHOLD and final < LOSS_RELATIVE_THRESHOLD * losses[0]
        assert not gate_would_have_passed, (
            f"the swapped-lengths run reached the gate's passing threshold "
            f"(final={final:.4f}) -- the gate would NOT have caught a "
            f"real swapped-lengths bug, which defeats its purpose."
        )


# --- (4) fresh DayLayers is a no-op -------------------------------------


def test_fresh_day_layers_leaves_input_unchanged() -> None:
    day_layers = DayLayers(n_sessions=3, n_features=8)
    x = torch.randn(4, 10, 8)
    session_indices = torch.tensor([0, 1, 2, 0])
    out = day_layers(x, session_indices)
    assert torch.allclose(out, x, atol=1e-6)


# --- (5) forward returns (time, batch, n_classes) -----------------------


def test_forward_returns_time_first_with_correct_class_dim() -> None:
    model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=32, use_day_layers=False)
    features = torch.randn(3, 20, 512)
    session_indices = torch.zeros(3, dtype=torch.int64)
    log_probs = model(features, session_indices)
    assert log_probs.shape == (20, 3, N_CLASSES)


# --- (6) log_softmax was applied over the class dimension ---------------


def test_output_rows_sum_to_one_after_exp() -> None:
    model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=32, use_day_layers=False)
    features = torch.randn(2, 15, 512)
    session_indices = torch.zeros(2, dtype=torch.int64)
    log_probs = model(features, session_indices)
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)


# --- (7) use_day_layers=False has no day-layer parameters ----------------


def test_use_day_layers_false_has_no_day_layer_parameters() -> None:
    model = GRUCTC(n_sessions=5, n_layers=1, hidden_size=32, use_day_layers=False)
    assert model.day_layers is None
    assert model.n_parameters()["day_layers"] == 0


# --- (8) ctc_input_lengths is currently identity --------------------------


def test_ctc_input_lengths_is_currently_identity() -> None:
    model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=32, use_day_layers=False)
    lengths = torch.tensor([10, 20, 5], dtype=torch.int64)
    out = model.ctc_input_lengths(lengths)
    assert torch.equal(out, lengths)


# --- (9) Gaussian smoothing: sigma=0 is a no-op; sigma=2 denoises --------


def test_smoothing_sigma_zero_is_a_no_op() -> None:
    model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=32, use_day_layers=False, smoothing_sigma_bins=0.0)
    x = torch.randn(2, 50, 512)
    assert torch.allclose(model._smooth(x), x)


def test_smoothing_sigma_two_reduces_white_noise_variance() -> None:
    model = GRUCTC(n_sessions=1, n_layers=1, hidden_size=32, use_day_layers=False, smoothing_sigma_bins=2.0)
    torch.manual_seed(0)
    x = torch.randn(4, 200, 512)
    smoothed = model._smooth(x)
    assert smoothed.var().item() < x.var().item()


# --- (10) freeze_all_except masks gradient to only that session ---------


def test_freeze_all_except_masks_gradient_to_only_that_session() -> None:
    day_layers = DayLayers(n_sessions=3, n_features=4)
    day_layers.freeze_all_except(1)

    x = torch.randn(3, 5, 4)
    session_indices = torch.tensor([0, 1, 2])
    out = day_layers(x, session_indices)
    out.sum().backward()

    assert day_layers.weights.grad is not None
    for s in range(3):
        if s == 1:
            assert day_layers.weights.grad[s].abs().sum() > 0
            assert day_layers.biases.grad[s].abs().sum() > 0
        else:
            assert torch.all(day_layers.weights.grad[s] == 0)
            assert torch.all(day_layers.biases.grad[s] == 0)
