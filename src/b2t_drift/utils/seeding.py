"""Deterministic seeding across random, numpy, and torch/torch.cuda."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs used in this project for reproducibility.

    Seeds Python's `random`, `numpy`, `torch` (CPU), and `torch.cuda` (all
    devices, if available). Optionally also configures cuDNN to use
    deterministic algorithms, which can slow down training but removes a
    source of run-to-run variance.

    Args:
        seed: Seed value to use for all RNGs.
        deterministic: If True, set `torch.backends.cudnn.deterministic =
            True` and `torch.backends.cudnn.benchmark = False`. If False,
            cuDNN is left free to pick faster, non-deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
