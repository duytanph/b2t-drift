"""Tests that temporal splits never leak sessions across time.

Not yet implemented: depends on `data/splits.py`, which is a stub.
"""

from __future__ import annotations


def test_no_session_appears_in_more_than_one_split() -> None:
    raise NotImplementedError


def test_train_sessions_all_precede_test_sessions() -> None:
    """Every train-split session date must be earlier than every
    test-split session date -- this is a temporal generalization study,
    not an i.i.d. one.
    """
    raise NotImplementedError
