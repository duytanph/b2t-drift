"""Tests for b2t_drift.phonemes."""

from __future__ import annotations

import pytest

from b2t_drift.phonemes import (
    LOGIT_TO_PHONEME,
    N_CLASSES,
    PHONEME_TO_LOGIT,
    decode_ids,
)


def test_length_is_41() -> None:
    assert len(LOGIT_TO_PHONEME) == 41
    assert N_CLASSES == 41


def test_index_0_is_blank() -> None:
    assert LOGIT_TO_PHONEME[0] == "BLANK"


def test_index_40_is_silence_token_exact() -> None:
    assert LOGIT_TO_PHONEME[40] == " | "


def test_round_trip_all_indices() -> None:
    for i, phoneme in enumerate(LOGIT_TO_PHONEME):
        assert PHONEME_TO_LOGIT[phoneme] == i


def test_decode_ids_raises_on_negative_index() -> None:
    with pytest.raises(ValueError):
        decode_ids([-1])


def test_decode_ids_raises_on_index_41() -> None:
    with pytest.raises(ValueError):
        decode_ids([41])


def test_all_entries_unique() -> None:
    assert len(set(LOGIT_TO_PHONEME)) == len(LOGIT_TO_PHONEME)
