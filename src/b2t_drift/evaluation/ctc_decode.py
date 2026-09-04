"""CTC decoding: turning model logits into phoneme sequences.

Responsible for greedy (argmax + collapse-repeats + remove-blanks) decoding
of CTC logits. Beam search is out of scope unless later needed.
"""

from __future__ import annotations

import torch


def greedy_ctc_decode(
    logits: torch.Tensor,
    blank_index: int = 0,
    lengths: torch.Tensor | None = None,
) -> list[list[int]]:
    """Greedily decode a batch of CTC logits into label-index sequences.

    Collapses repeated consecutive predictions and removes blank tokens,
    per the standard CTC greedy decoding procedure. Works identically
    whether `logits` are raw logits or log-probabilities (e.g.
    `GRUCTC.forward`'s output, transposed to batch-first) -- `argmax` is
    invariant under `log_softmax`, which is a monotonic, per-row
    transform.

    Args:
        logits: Per-timestep class scores, shape `(batch, time,
            vocab_size)`. NOTE: this is BATCH-FIRST, the opposite of
            `GRUCTC.forward`'s time-first return -- callers passing a
            model's output here must `.transpose(0, 1)` it first.
        blank_index: Index of the CTC blank token.
        lengths: `(batch,)`, each item's true (unpadded) time length. If
            given, timesteps at or past `lengths[i]` are excluded from
            item `i`'s decode before collapsing -- padded regions have no
            defined "correct" prediction and including them risks
            appending spurious trailing symbols to the decoded sequence.
            If None (the default), every item is decoded over its full
            `time` extent; only correct when the batch has no padding
            (e.g. `batch_size=1`, or every item already truncated to its
            own length).

    Returns:
        One decoded label-index sequence per batch element, with
        consecutive repeats collapsed and `blank_index` removed.
    """
    predictions = logits.argmax(dim=-1)  # (batch, time)
    batch_size, max_time = predictions.shape

    decoded: list[list[int]] = []
    for b in range(batch_size):
        t = int(lengths[b]) if lengths is not None else max_time
        seq = predictions[b, :t].tolist()

        collapsed = []
        prev = None
        for label in seq:
            if label != prev:
                collapsed.append(label)
            prev = label
        decoded.append([label for label in collapsed if label != blank_index])

    return decoded
