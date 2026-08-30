"""Per-channel normalization statistics for neural features.

Responsible for computing (e.g. median/MAD or mean/std) normalization
statistics from training data and applying them at load time. Statistics
must be computed only from training-split sessions to avoid leakage.
"""

from __future__ import annotations

import torch


class NormalizationStats:
    """Per-channel normalization statistics for neural feature arrays.

    TODO: decide whether statistics are computed per-session or globally
    across the training split, once drift behavior is better understood.
    """

    def __init__(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        """Store precomputed per-channel center and scale values.

        Args:
            center: Per-channel center (e.g. median or mean), shape
                `(num_channels,)`.
            scale: Per-channel scale (e.g. MAD or std), shape
                `(num_channels,)`.
        """
        raise NotImplementedError

    @classmethod
    def from_examples(cls, examples: list[torch.Tensor]) -> "NormalizationStats":
        """Compute normalization statistics from a list of training examples.

        Args:
            examples: Neural feature tensors from the training split only.

        Returns:
            A fitted `NormalizationStats` instance.
        """
        raise NotImplementedError

    def apply(self, neural: torch.Tensor) -> torch.Tensor:
        """Normalize a neural feature tensor using the stored statistics.

        Args:
            neural: Neural feature tensor, shape `(..., num_channels)`.

        Returns:
            Normalized neural feature tensor, same shape as input.
        """
        raise NotImplementedError
