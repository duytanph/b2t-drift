"""GRU + CTC phoneme decoder.

The single model family used in this project: a GRU encoder over neural
features, followed by a linear classifier trained with CTC loss against
phoneme label sequences. No other architectures are planned -- see
CLAUDE.md on avoiding premature abstraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUCTCDecoder(nn.Module):
    """GRU encoder with a linear CTC output head.

    TODO: finalize input feature dimensionality (512, per project
    description) and output vocabulary size once `phonemes.py` is
    implemented.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        """Build the GRU/CTC decoder.

        Args:
            input_size: Number of input neural feature channels.
            hidden_size: GRU hidden state size.
            num_layers: Number of stacked GRU layers.
            vocab_size: Output vocabulary size, including the blank token.
            dropout: Dropout between GRU layers.
            bidirectional: Whether the GRU is bidirectional.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep class logits.

        Args:
            x: Neural feature tensor, shape `(batch, time, input_size)`.
            lengths: Valid sequence length per batch element.

        Returns:
            Logits, shape `(batch, time, vocab_size)`.
        """
        raise NotImplementedError
