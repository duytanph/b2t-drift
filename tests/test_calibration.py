"""Tests for `b2t_drift.evaluation.calibration` -- Phase 3's calibration/
evaluation trial partitioning utility.
"""

from __future__ import annotations

from datetime import date

import pytest

from b2t_drift.data.hdf5_io import TrialId
from b2t_drift.evaluation.calibration import (
    calibration_minutes,
    calibration_split_seed,
    normalize_sentence,
    split_calibration_trials,
    split_off_sentence_overlap,
)


def _trial(trial_key: str, n_time_steps: int = 100, seq_len: int = 5) -> TrialId:
    return TrialId(
        session_id="t15.2024.02.25",
        session_date=date(2024, 2, 25),
        partition="train",
        trial_key=trial_key,
        block_num=1,
        trial_num=0,
        n_time_steps=n_time_steps,
        seq_len=seq_len,
        sentence_label="synthetic",
    )


def _trials(n: int) -> list[TrialId]:
    return [_trial(f"trial_{i:04d}") for i in range(n)]


def test_split_partitions_exactly_with_no_overlap():
    trials = _trials(20)
    calib, evald = split_calibration_trials(trials, n_calibration=5, seed=0)
    assert len(calib) == 5
    assert len(evald) == 15
    assert set(calib) | set(evald) == set(trials)
    assert set(calib) & set(evald) == set()


def test_split_is_deterministic_given_same_seed():
    trials = _trials(30)
    calib1, eval1 = split_calibration_trials(trials, n_calibration=10, seed=42)
    calib2, eval2 = split_calibration_trials(trials, n_calibration=10, seed=42)
    assert calib1 == calib2
    assert eval1 == eval2


def test_different_seeds_give_different_partitions():
    trials = _trials(30)
    calib1, _ = split_calibration_trials(trials, n_calibration=10, seed=1)
    calib2, _ = split_calibration_trials(trials, n_calibration=10, seed=2)
    assert set(calib1) != set(calib2)


def test_preserves_relative_order_within_each_group():
    trials = _trials(10)
    calib, evald = split_calibration_trials(trials, n_calibration=4, seed=7)
    original_order = {t: i for i, t in enumerate(trials)}
    assert [original_order[t] for t in calib] == sorted(original_order[t] for t in calib)
    assert [original_order[t] for t in evald] == sorted(original_order[t] for t in evald)


def test_n_calibration_zero_is_arm_0():
    trials = _trials(10)
    calib, evald = split_calibration_trials(trials, n_calibration=0, seed=0)
    assert calib == []
    assert evald == trials


def test_impossible_cell_raises_not_silently_reduced():
    trials = _trials(10)
    with pytest.raises(ValueError, match="IMPOSSIBLE"):
        split_calibration_trials(trials, n_calibration=10, seed=0)
    with pytest.raises(ValueError, match="IMPOSSIBLE"):
        split_calibration_trials(trials, n_calibration=15, seed=0)


def test_negative_n_calibration_raises():
    trials = _trials(10)
    with pytest.raises(ValueError, match="n_calibration"):
        split_calibration_trials(trials, n_calibration=-1, seed=0)


def test_calibration_split_seed_is_deterministic():
    s1 = calibration_split_seed("t15.2024.02.25", 10, 0)
    s2 = calibration_split_seed("t15.2024.02.25", 10, 0)
    assert s1 == s2


def test_calibration_split_seed_varies_with_each_input():
    base = calibration_split_seed("t15.2024.02.25", 10, 0)
    assert base != calibration_split_seed("t15.2024.02.25", 10, 1)  # split_index
    assert base != calibration_split_seed("t15.2024.02.25", 25, 0)  # n_calibration
    assert base != calibration_split_seed("t15.2024.07.19", 10, 0)  # session_id
    assert base != calibration_split_seed("t15.2024.02.25", 10, 0, base_seed=1)  # base_seed


def test_same_cell_seed_reproduces_same_partition_across_two_call_sites():
    """The whole point: two independent callers (e.g. two different arms'
    code paths) asking for the same (session, N, split) must land on the
    identical partition without passing trial lists to each other."""
    trials = _trials(50)
    seed_a = calibration_split_seed("t15.2024.02.25", 25, 3)
    seed_b = calibration_split_seed("t15.2024.02.25", 25, 3)
    calib_a, eval_a = split_calibration_trials(trials, 25, seed_a)
    calib_b, eval_b = split_calibration_trials(trials, 25, seed_b)
    assert calib_a == calib_b
    assert eval_a == eval_b


def test_calibration_minutes_sums_time_steps_at_20ms_bins():
    trials = [_trial("a", n_time_steps=3000), _trial("b", n_time_steps=1500)]
    # (3000 + 1500) * 20ms = 90,000 ms = 90 s = 1.5 min
    assert calibration_minutes(trials) == pytest.approx(1.5)


def test_calibration_minutes_empty_is_zero():
    assert calibration_minutes([]) == 0.0


def test_calibration_minutes_custom_bin_ms():
    trials = [_trial("a", n_time_steps=100)]
    assert calibration_minutes(trials, bin_ms=10.0) == pytest.approx(100 * 10.0 / 1000.0 / 60.0)


# --- sentence-level de-duplication -----------------------------------------


def _labeled(trial_key: str, sentence: str) -> TrialId:
    return TrialId(
        session_id="t15.2024.07.19", session_date=date(2024, 7, 19), partition="train",
        trial_key=trial_key, block_num=1, trial_num=0, n_time_steps=100, seq_len=5,
        sentence_label=sentence,
    )


def _trials_with_duplicated_sentences() -> list[TrialId]:
    """10 distinct trials but only 5 distinct sentences: every sentence
    appears exactly twice, as two DIFFERENT trials. Tests below calibrate on an
    ODD number of trials (5): an even number could pick complete pairs and
    produce no overlap at all, while an odd number guarantees at least one
    calibrated sentence whose twin lands in evaluation."""
    return [_labeled(f"trial_{i:04d}", f"sentence number {i % 5}.") for i in range(10)]


def test_normalize_sentence_folds_case_and_whitespace_only():
    assert normalize_sentence("  You CAN   go. ") == "you can go."
    assert normalize_sentence("you can go.") == normalize_sentence("You can go.")
    assert normalize_sentence("you can go") != normalize_sentence("you can go.")  # punctuation kept


def test_split_off_sentence_overlap_removes_only_matching_eval_trials():
    calib = [_labeled("c0", "hello there."), _labeled("c1", "another one.")]
    evald = [_labeled("e0", "Hello  there."), _labeled("e1", "something new."), _labeled("e2", "another one.")]
    kept, removed = split_off_sentence_overlap(calib, evald)
    assert [t.trial_key for t in kept] == ["e1"]
    assert [t.trial_key for t in removed] == ["e0", "e2"]


def test_default_split_keeps_duplicate_sentences_in_eval():
    """The default is unchanged (trial-level only) -- so the leak the
    Phase 3 grid had is reproducible, not silently altered."""
    trials = _trials_with_duplicated_sentences()
    calib, evald = split_calibration_trials(trials, n_calibration=5, seed=3)
    assert len(calib) + len(evald) == 10
    cal_sents = {t.sentence_label for t in calib}
    assert any(t.sentence_label in cal_sents for t in evald)  # overlap exists and is kept


def test_dedup_sentences_removes_every_calibrated_sentence_from_eval():
    trials = _trials_with_duplicated_sentences()
    calib, evald = split_calibration_trials(trials, n_calibration=5, seed=3, dedup_sentences=True)
    cal_sents = {normalize_sentence(t.sentence_label) for t in calib}
    assert all(normalize_sentence(t.sentence_label) not in cal_sents for t in evald)
    assert set(calib) & set(evald) == set()


def test_dedup_leaves_the_calibration_set_unchanged():
    trials = _trials_with_duplicated_sentences()
    calib_plain, eval_plain = split_calibration_trials(trials, n_calibration=5, seed=3)
    calib_dedup, eval_dedup = split_calibration_trials(trials, n_calibration=5, seed=3, dedup_sentences=True)
    assert calib_plain == calib_dedup  # same seed -> same calibration trials either way
    assert set(eval_dedup) <= set(eval_plain)  # dedup'd eval is a strict subset
    assert len(eval_dedup) < len(eval_plain)


def test_dedup_is_a_noop_when_all_sentences_are_distinct():
    trials = [_labeled(f"trial_{i:04d}", f"unique sentence {i}.") for i in range(10)]
    plain = split_calibration_trials(trials, n_calibration=4, seed=1)
    dedup = split_calibration_trials(trials, n_calibration=4, seed=1, dedup_sentences=True)
    assert plain == dedup
