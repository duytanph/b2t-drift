"""Tests for b2t_drift.evaluation.ctc_forward_backward (synthetic data only)."""

from __future__ import annotations

import torch

from b2t_drift.evaluation.ctc_forward_backward import (
    ctc_forward_backward,
    map_posteriors_to_bins,
)

BLANK = 0


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=-1)


def test_posteriors_sum_to_one_per_timestep() -> None:
    torch.manual_seed(0)
    t_steps, n_classes = 20, 5
    log_probs = _log_softmax(torch.randn(t_steps, n_classes, dtype=torch.float64))
    target = torch.tensor([1, 2, 3], dtype=torch.int64)

    posteriors, _ = ctc_forward_backward(log_probs, target, blank=BLANK)

    row_sums = posteriors.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(t_steps, dtype=torch.float64), atol=1e-8)


def test_forced_alignment_when_t_equals_target_length_no_repeats() -> None:
    """When T == L (the RAW target length, not the extended length S) and
    no two consecutive target labels repeat, CTC has EXACTLY ONE valid
    path: skip every blank and land on each label in turn (average
    progress of +2 states/timestep is required to cover the 2L-2 states
    between the first and last label positions in only L-1 transitions,
    which leaves no slack for a "stay" or a "+1 advance onto a blank"
    anywhere). So the posterior at timestep t must put (essentially) all
    its mass on the t-th TARGET LABEL, regardless of what the raw
    log_probs say (as long as no class is assigned exactly log(0)) -- a
    "known alignment" the combinatorics force, not one obtained by
    hand-crafting peaked log_probs.
    """
    torch.manual_seed(1)
    target = torch.tensor([1, 2, 3], dtype=torch.int64)  # no consecutive repeats
    t_steps, n_classes = 3, 5  # T == L
    log_probs = _log_softmax(torch.randn(t_steps, n_classes, dtype=torch.float64) * 3)

    posteriors, _ = ctc_forward_backward(log_probs, target, blank=BLANK)

    for t, expected_class in enumerate(target.tolist()):
        assert posteriors[t, expected_class].item() > 0.999999, (
            f"t={t}: expected forced posterior ~1.0 on class {expected_class}, "
            f"got {posteriors[t].tolist()}"
        )


def test_log_likelihood_matches_nn_ctc_loss() -> None:
    """The real correctness check: forward-backward's marginal
    log-likelihood must match nn.CTCLoss's (negated) value for the exact
    same (log_probs, target) -- an independent implementation computing
    the same quantity a different way."""
    torch.manual_seed(2)
    t_steps, n_classes = 30, 6
    target = torch.tensor([1, 2, 1, 3, 4], dtype=torch.int64)

    logits = torch.randn(t_steps, n_classes, dtype=torch.float64) * 2
    log_probs = _log_softmax(logits)

    _, log_likelihood = ctc_forward_backward(log_probs, target, blank=BLANK)

    ctc_loss_value = torch.nn.functional.ctc_loss(
        log_probs.unsqueeze(1).float(),  # (T, N=1, C), nn.CTCLoss wants batch dim
        target.unsqueeze(0),
        input_lengths=torch.tensor([t_steps]),
        target_lengths=torch.tensor([target.shape[0]]),
        blank=BLANK,
        reduction="none",
    )
    # nn.CTCLoss returns NEGATIVE log likelihood; ctc_forward_backward
    # returns log likelihood directly.
    assert torch.allclose(
        log_likelihood, -ctc_loss_value.double()[0], atol=1e-3,
    ), f"log_likelihood={log_likelihood.item()}, -ctc_loss={-ctc_loss_value.item()}"


def test_log_likelihood_matches_nn_ctc_loss_empty_target() -> None:
    """Edge case: an empty target (S=1, blank-only) must still match
    nn.CTCLoss."""
    torch.manual_seed(3)
    t_steps, n_classes = 10, 4
    target = torch.zeros(0, dtype=torch.int64)
    log_probs = _log_softmax(torch.randn(t_steps, n_classes, dtype=torch.float64))

    _, log_likelihood = ctc_forward_backward(log_probs, target, blank=BLANK)

    ctc_loss_value = torch.nn.functional.ctc_loss(
        log_probs.unsqueeze(1).float(),
        target.unsqueeze(0),
        input_lengths=torch.tensor([t_steps]),
        target_lengths=torch.tensor([0]),
        blank=BLANK,
        reduction="none",
    )
    assert torch.allclose(log_likelihood, -ctc_loss_value.double()[0], atol=1e-3)


def test_posteriors_concentrate_near_correct_labels_when_log_probs_peaked() -> None:
    """A less combinatorially-forced case, WITH slack (T > L): log_probs
    strongly favor one specific, explicitly-validated-as-legal alignment
    path at each timestep -- posteriors should concentrate there too,
    since any competing path must hit at least one timestep's strongly
    dis-favored class.

    The state path below (indices into the extended sequence
    z = [blank, 1, blank, 2, blank, 3, blank], S=7) is checked for
    validity by construction: monotonic, every step is a stay (0), an
    advance (+1), or nothing larger -- never an illegal skip -- so it is
    guaranteed realizable, unlike hand-picking class labels per timestep
    (which silently assumed a state trajectory that turned out not to be
    uniquely valid in an earlier version of this test).
    """
    target = torch.tensor([1, 2, 3], dtype=torch.int64)
    z = [0, 1, 0, 2, 0, 3, 0]  # extended sequence's class at each state index
    state_path = [0, 0, 1, 1, 2, 3, 4, 5, 6, 6]  # monotonic, steps of 0 or +1 only
    for a, b in zip(state_path, state_path[1:]):
        assert b - a in (0, 1), "state_path must only stay or advance by 1 for this construction"

    expected_classes = [z[s] for s in state_path]
    t_steps, n_classes = len(state_path), 5
    logits = torch.full((t_steps, n_classes), -5.0, dtype=torch.float64)
    for t, c in enumerate(expected_classes):
        logits[t, c] = 10.0
    log_probs = _log_softmax(logits)

    posteriors, _ = ctc_forward_backward(log_probs, target, blank=BLANK)

    for t, c in enumerate(expected_classes):
        assert posteriors[t, c].item() > 0.9, f"t={t}: expected concentration on class {c}, got {posteriors[t].tolist()}"


def test_raises_when_fewer_timesteps_than_raw_target_length() -> None:
    target = torch.tensor([1, 2, 3, 4], dtype=torch.int64)  # L = 4
    log_probs = _log_softmax(torch.randn(3, 5, dtype=torch.float64))  # T = 3 < 4
    try:
        ctc_forward_backward(log_probs, target, blank=BLANK)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "T=3" in str(exc) and "L=4" in str(exc)


def test_does_not_raise_when_t_below_extended_length_but_above_target_length() -> None:
    """T between L and S (exclusive) is legal whenever the target has no
    consecutive repeated labels -- see the module's `Raises` docstring."""
    target = torch.tensor([1, 2, 3], dtype=torch.int64)  # L=3, S=7
    log_probs = _log_softmax(torch.randn(4, 5, dtype=torch.float64))  # T=4: L < T < S
    posteriors, log_likelihood = ctc_forward_backward(log_probs, target, blank=BLANK)
    assert torch.isfinite(log_likelihood)
    assert torch.allclose(posteriors.sum(dim=1), torch.ones(4, dtype=torch.float64), atol=1e-8)


def test_map_posteriors_to_bins_repeats_each_frame_by_stride() -> None:
    posteriors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    mapped = map_posteriors_to_bins(posteriors, patch_stride=4)
    assert mapped.shape == (12, 2)
    assert torch.equal(mapped[0:4], posteriors[0].expand(4, -1))
    assert torch.equal(mapped[4:8], posteriors[1].expand(4, -1))
    assert torch.equal(mapped[8:12], posteriors[2].expand(4, -1))
