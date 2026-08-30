"""PyTorch Dataset for T15 neural/phoneme sequence pairs.

Wraps `hdf5_io` to expose per-trial (neural_features, phoneme_labels)
examples via the standard `torch.utils.data.Dataset` interface.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class T15Dataset(Dataset):
    """A single session (or set of sessions) of neural/phoneme examples.

    TODO: decide whether this dataset spans one session or many, and how
    session identity is exposed to downstream code (e.g. for day layers),
    once the data has been inspected.
    """

    def __init__(self, session_paths: list[Path]) -> None:
        """Build the dataset from a list of session HDF5 file paths.

        Args:
            session_paths: Paths to session HDF5 files to include.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Total number of trials across all included sessions."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch one (neural_features, phoneme_labels) example.

        Args:
            idx: Example index.

        Returns:
            A tuple of (neural_features, phoneme_labels) tensors.
        """
        raise NotImplementedError
