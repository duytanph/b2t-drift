"""Batch collation for variable-length neural/phoneme sequences.

Responsible for padding a list of (neural_features, phoneme_labels)
examples into batched tensors suitable for `nn.GRU` and `nn.CTCLoss`.
"""

from __future__ import annotations

import torch


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad and stack a list of variable-length examples into a batch.

    Args:
        batch: A list of (neural_features, phoneme_labels) tensor pairs,
            as returned by `T15Dataset.__getitem__`.

    Returns:
        A tuple of (padded_neural, padded_labels, neural_lengths,
        label_lengths), suitable for `nn.utils.rnn.pack_padded_sequence`
        and `nn.CTCLoss`.
    """
    raise NotImplementedError
