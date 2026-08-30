"""Tests for b2t_drift.config."""

from __future__ import annotations

from pathlib import Path

from b2t_drift.config import load_data_config
from b2t_drift.phonemes import N_CLASSES

T15_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "data" / "t15.yaml"


def test_t15_ctc_output_size_matches_phoneme_vocabulary() -> None:
    config = load_data_config(T15_CONFIG_PATH)
    assert config.ctc_output_size == N_CLASSES
