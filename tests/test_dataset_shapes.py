"""Tests that T15Dataset examples have the expected tensor shapes.

Not yet implemented: requires a small fixture HDF5 file, which does not
exist until the real T15 HDF5 structure has been inspected.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="stub: module not yet implemented")
def test_example_shapes_match_config() -> None:
    """A dataset example's neural tensor should be (time, num_features)."""
    raise NotImplementedError


@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="stub: module not yet implemented")
def test_collated_batch_shapes() -> None:
    """A collated batch should pad to (batch, max_time, num_features)."""
    raise NotImplementedError
