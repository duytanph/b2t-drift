"""Per-session ("day") input adaptation layers.

Responsible for any session-specific transformation applied to neural
features before the shared decoder, used to compensate for
session-to-session shifts in neural statistics.

TODO: confirm whether this project uses day layers at all, and if so,
their exact form (affine, learned embedding, etc.) before implementing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DayLayer(nn.Module):
    """A single session-specific input transformation.

    TODO: parameterization not yet decided.
    """

    def __init__(self, num_channels: int, session_id: str) -> None:
        """Initialize a day layer for one session.

        Args:
            num_channels: Number of input neural feature channels.
            session_id: Identifier of the session this layer adapts to.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the session-specific transformation.

        Args:
            x: Neural feature tensor, shape `(batch, time, num_channels)`.

        Returns:
            Transformed neural feature tensor, same shape as input.
        """
        raise NotImplementedError


class DayLayerBank(nn.Module):
    """A collection of `DayLayer`s indexed by session id.

    TODO: decide lookup/dispatch mechanism (dict of modules vs. embedding
    table) once day-layer design is finalized.
    """

    def __init__(self, num_channels: int, session_ids: list[str]) -> None:
        """Initialize one day layer per session id.

        Args:
            num_channels: Number of input neural feature channels.
            session_ids: All session ids the bank should have a layer for.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, session_id: str) -> torch.Tensor:
        """Apply the day layer for a given session.

        Args:
            x: Neural feature tensor, shape `(batch, time, num_channels)`.
            session_id: Which session's day layer to apply.

        Returns:
            Transformed neural feature tensor, same shape as input.
        """
        raise NotImplementedError
