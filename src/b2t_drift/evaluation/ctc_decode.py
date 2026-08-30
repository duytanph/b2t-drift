"""CTC decoding: turning model logits into phoneme sequences.

Responsible for greedy (argmax + collapse-repeats + remove-blanks) decoding
of CTC logits. Beam search is out of scope unless later needed.
"""

from __future__ import annotations

import torch


def greedy_ctc_decode(logits: torch.Tensor, blank_index: int = 0) -> list[list[int]]:
    """Greedily decode a batch of CTC logits into label-index sequences.

    Collapses repeated consecutive predictions and removes blank tokens,
    per the standard CTC greedy decoding procedure.

    Args:
        logits: Per-timestep class logits, shape `(batch, time, vocab_size)`.
        blank_index: Index of the CTC blank token.

    Returns:
        One decoded label-index sequence per batch element.
    """
    raise NotImplementedError
