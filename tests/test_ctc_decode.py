"""Unit tests for `b2t_drift.evaluation.ctc_decode.greedy_ctc_decode`.

Targets Task 3: `lengths` is now required (not an opt-out default), plus
the three assertions that catch a mis-shaped or axis-swapped `logits`
tensor before it can silently produce garbage decodes, and the
silence-token decision (silence is scored, blanks are not).
"""

from __future__ import annotations

import pytest
import torch

from b2t_drift.evaluation.ctc_decode import greedy_ctc_decode

BLANK = 0
SILENCE = 40
N_CLASSES = 41


def _one_hot_logits(class_sequence: list[int], n_classes: int = N_CLASSES) -> torch.Tensor:
    """Build (1, time, n_classes) logits whose argmax is exactly `class_sequence`."""
    time = len(class_sequence)
    logits = torch.full((1, time, n_classes), -10.0)
    for t, c in enumerate(class_sequence):
        logits[0, t, c] = 10.0
    return logits


# --- basic correctness -------------------------------------------------


def test_collapses_repeats_and_removes_blank() -> None:
    logits = _one_hot_logits([5, 5, BLANK, 5, 7, 7, BLANK])
    lengths = torch.tensor([7])
    decoded = greedy_ctc_decode(logits, lengths)
    assert decoded == [[5, 5, 7]]


def test_silence_is_scored_not_removed() -> None:
    logits = _one_hot_logits([5, SILENCE, SILENCE, 7])
    lengths = torch.tensor([4])
    decoded = greedy_ctc_decode(logits, lengths)
    # silence collapses like any other repeated class, but is NOT dropped.
    assert decoded == [[5, SILENCE, 7]]


def test_lengths_truncates_padded_region() -> None:
    # Real content is [5, 7]; timesteps past length=2 are padding garbage
    # that must not leak into the decode.
    logits = _one_hot_logits([5, 7, 3, 3, 3])
    lengths = torch.tensor([2])
    decoded = greedy_ctc_decode(logits, lengths)
    assert decoded == [[5, 7]]


def test_batched_decode_with_different_lengths() -> None:
    a = _one_hot_logits([5, 7, BLANK, 9, 9])
    b = _one_hot_logits([3, 3, BLANK, 1, 1])
    logits = torch.cat([a, b], dim=0)
    lengths = torch.tensor([3, 5])
    decoded = greedy_ctc_decode(logits, lengths)
    assert decoded == [[5, 7], [3, 1]]


# --- Task 3(a): required lengths + assertions ---------------------------


def test_lengths_is_a_required_argument() -> None:
    logits = _one_hot_logits([5, 7])
    with pytest.raises(TypeError):
        greedy_ctc_decode(logits)  # type: ignore[call-arg]


def test_raises_on_non_3d_logits() -> None:
    logits = torch.randn(4, 41)  # missing the batch or time dim
    lengths = torch.tensor([4])
    with pytest.raises(AssertionError, match="ndim"):
        greedy_ctc_decode(logits, lengths)


def test_axis_order_mismatch_is_caught() -> None:
    """THE case this function used to let through silently: passing
    GRUCTC.forward's (time, batch, n_classes) output un-transposed."""
    time, batch = 50, 4
    logits_time_first = torch.randn(time, batch, N_CLASSES)  # NOT transposed
    lengths = torch.tensor([10, 12, 8, 20])  # length batch_size=4

    with pytest.raises(AssertionError, match="shape\\[0\\]"):
        greedy_ctc_decode(logits_time_first, lengths)


def test_n_classes_mismatch_is_caught() -> None:
    logits = torch.randn(2, 10, 30)  # wrong vocab size
    lengths = torch.tensor([10, 10])
    with pytest.raises(AssertionError, match="n_classes"):
        greedy_ctc_decode(logits, lengths, n_classes=41)


def test_n_classes_matches_by_default_for_this_projects_vocabulary() -> None:
    logits = _one_hot_logits([1, 2, 3])
    lengths = torch.tensor([3])
    # Should not raise -- default n_classes=41 matches _one_hot_logits'
    # default width.
    greedy_ctc_decode(logits, lengths)
