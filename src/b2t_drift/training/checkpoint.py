"""Model checkpointing.

Responsible for saving and loading model/optimizer state to `runs/`, along
with the metadata needed to reproduce a run (config, seed, git commit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metadata: dict[str, Any],
) -> None:
    """Save model/optimizer state and run metadata to disk.

    Args:
        path: Destination checkpoint file path, under `runs/`.
        model: Model whose state should be saved.
        optimizer: Optimizer whose state should be saved.
        epoch: Current epoch number.
        metadata: Additional run metadata (e.g. config, seed, git commit).
    """
    raise NotImplementedError


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load model (and optionally optimizer) state from a checkpoint file.

    Args:
        path: Checkpoint file path to load.
        model: Model to load state into, in place.
        optimizer: Optional optimizer to load state into, in place.

    Returns:
        The checkpoint's stored metadata dict.
    """
    raise NotImplementedError
