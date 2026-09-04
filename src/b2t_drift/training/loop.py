"""Training and validation loop.

Responsible for the epoch-level train/eval loop: iterating batches,
computing CTC loss, stepping the optimizer, and logging metrics. Called
by `scripts/02_train.py`; contains no CLI or config-parsing logic itself.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Args:
        model: The model being trained.
        dataloader: Yields batches as produced by `collate_trials`.
        optimizer: Optimizer for `model`'s parameters.
        criterion: Loss function (expected to be `nn.CTCLoss`).
        device: Device to run computation on.

    Returns:
        Average training loss over the epoch.
    """
    raise NotImplementedError


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Run evaluation over a dataloader without updating model weights.

    Args:
        model: The model being evaluated.
        dataloader: Yields batches as produced by `collate_trials`.
        criterion: Loss function (expected to be `nn.CTCLoss`).
        device: Device to run computation on.

    Returns:
        A dict of evaluation metrics (e.g. loss, phoneme error rate).
    """
    raise NotImplementedError
