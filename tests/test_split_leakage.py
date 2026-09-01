"""Tests that temporal splits never leak sessions across time.

Not yet implemented: depends on `data/splits.py`, which is a stub.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="stub: module not yet implemented")
def test_no_session_appears_in_more_than_one_split() -> None:
    raise NotImplementedError


@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="stub: module not yet implemented")
def test_train_sessions_all_precede_test_sessions() -> None:
    """Every train-split session date must be earlier than every
    test-split session date -- this is a temporal generalization study,
    not an i.i.d. one.
    """
    raise NotImplementedError
