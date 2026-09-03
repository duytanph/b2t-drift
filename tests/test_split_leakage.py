"""Tests that temporal splits never leak sessions or trials across time.

Run against the real T15 data files -- the whole module is skipped if
data/raw/hdf5_data_final is not present.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import pytest

from b2t_drift.data.hdf5_io import parse_session_date
from b2t_drift.data.splits import (
    build_temporal_split,
    load_corpus_labels,
    load_split,
    split_to_trial_ids,
    validate_split,
    write_split,
)

DATA_ROOT = Path("data/raw/hdf5_data_final")
CORPUS_CSV = Path("data/external/t15_copyTaskData_description.csv")


@pytest.fixture(autouse=True)
def _require_real_data() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"real data not present at {DATA_ROOT}")
    if not CORPUS_CSV.exists():
        pytest.skip(f"corpus CSV not present at {CORPUS_CSV}")


@pytest.fixture(scope="module")
def sb_split():
    """The base-case Switchboard-only split, built once per test module."""
    return build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=15, corpus_filter="Switchboard")


def _all_trial_refs(split_dict: dict) -> list[tuple]:
    refs = [tuple(r) for r in split_dict["train"]] + [tuple(r) for r in split_dict["val"]]
    for e in split_dict["eval_sessions"]:
        refs.extend(tuple(r) for r in e["trials"])
    return refs


def test_no_session_in_both_training_and_eval(sb_split) -> None:
    train_val_sessions = {r[0] for r in sb_split["train"]} | {r[0] for r in sb_split["val"]}
    eval_sessions = {e["session_id"] for e in sb_split["eval_sessions"]}
    assert train_val_sessions.isdisjoint(eval_sessions)


def test_every_training_session_date_le_cut_date(sb_split) -> None:
    cut_date = date.fromisoformat(sb_split["cut_date"])
    train_val_sessions = {r[0] for r in sb_split["train"]} | {r[0] for r in sb_split["val"]}
    for session_id in train_val_sessions:
        assert parse_session_date(session_id) <= cut_date


def test_every_eval_session_date_after_cut_date(sb_split) -> None:
    cut_date = date.fromisoformat(sb_split["cut_date"])
    for e in sb_split["eval_sessions"]:
        assert date.fromisoformat(e["session_date"]) > cut_date


def test_no_trial_reference_appears_in_more_than_one_partition(sb_split) -> None:
    refs = _all_trial_refs(sb_split)
    assert len(refs) == len(set(refs))


def test_train_and_val_trial_sets_are_disjoint(sb_split) -> None:
    train_set = {tuple(r) for r in sb_split["train"]}
    val_set = {tuple(r) for r in sb_split["val"]}
    assert train_set.isdisjoint(val_set)


def test_every_val_trials_session_is_a_training_session(sb_split) -> None:
    train_sessions = {r[0] for r in sb_split["train"]}
    val_sessions = {r[0] for r in sb_split["val"]}
    assert val_sessions <= train_sessions


def test_session_to_index_covers_every_referenced_session_contiguous(sb_split) -> None:
    referenced = {r[0] for r in sb_split["train"]} | {r[0] for r in sb_split["val"]}
    referenced |= {e["session_id"] for e in sb_split["eval_sessions"]}
    assert referenced <= set(sb_split["session_to_index"])

    values = sorted(sb_split["session_to_index"].values())
    assert values == list(range(len(values)))


def test_days_since_cut_matches_session_date_minus_cut_date(sb_split) -> None:
    cut_date = date.fromisoformat(sb_split["cut_date"])
    for e in sb_split["eval_sessions"]:
        expected = (date.fromisoformat(e["session_date"]) - cut_date).days
        assert e["days_since_cut"] == expected


def test_switchboard_filter_matches_corpus_csv(sb_split) -> None:
    corpus_labels = load_corpus_labels(CORPUS_CSV)
    all_refs = _all_trial_refs(sb_split)
    assert len(all_refs) > 0  # sanity: the fixture actually has trials

    # Need each ref's block_num, which trial references alone don't carry
    # -- rehydrate via split_to_trial_ids for each group instead of
    # per-trial re-reads.
    for trial_id in split_to_trial_ids(sb_split, DATA_ROOT, "train"):
        assert corpus_labels[(trial_id.session_id, trial_id.block_num)] == "Switchboard"
    for trial_id in split_to_trial_ids(sb_split, DATA_ROOT, "val"):
        assert corpus_labels[(trial_id.session_id, trial_id.block_num)] == "Switchboard"


def test_validate_split_raises_on_session_moved_into_eval(sb_split) -> None:
    corrupted = copy.deepcopy(sb_split)
    # Move one training session's trials into a fabricated eval entry.
    moved_session_id = corrupted["train"][0][0]
    moved_trials = [r for r in corrupted["train"] if r[0] == moved_session_id]
    corrupted["train"] = [r for r in corrupted["train"] if r[0] != moved_session_id]
    corrupted["eval_sessions"].append({
        "session_id": moved_session_id,
        "session_date": parse_session_date(moved_session_id).isoformat(),
        "days_since_cut": -1,
        "n_trials": len(moved_trials),
        "median_seq_len": 0.0,
        "passes_threshold": False,
        "trials": moved_trials,
    })

    with pytest.raises(ValueError):
        validate_split(corrupted)


def test_validate_split_raises_when_eval_session_precedes_cut_date(sb_split) -> None:
    corrupted = copy.deepcopy(sb_split)
    corrupted["eval_sessions"][0]["session_date"] = corrupted["cut_date"]  # == cut_date, not after

    with pytest.raises(ValueError):
        validate_split(corrupted)


def test_write_then_load_split_round_trips(sb_split, tmp_path) -> None:
    path = tmp_path / "roundtrip.json"
    write_split(sb_split, path)
    loaded = load_split(path)
    assert loaded == sb_split


def test_split_to_trial_ids_session_ids_match_manifest(sb_split) -> None:
    train_trial_ids = split_to_trial_ids(sb_split, DATA_ROOT, "train")
    expected_sessions = {r[0] for r in sb_split["train"]}
    assert {t.session_id for t in train_trial_ids} == expected_sessions
    assert len(train_trial_ids) == len(sb_split["train"])


def test_all_four_switchboard_cut_points_validate_cleanly() -> None:
    for cut in (5, 10, 15, 20):
        split = build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=cut, corpus_filter="Switchboard")
        validate_split(split)  # must not raise


def test_later_cut_points_have_more_train_trials_and_fewer_eval_sessions() -> None:
    splits = {
        cut: build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=cut, corpus_filter="Switchboard")
        for cut in (5, 10, 15, 20)
    }
    cuts = sorted(splits)
    for earlier, later in zip(cuts, cuts[1:]):
        assert len(splits[later]["train"]) > len(splits[earlier]["train"])
        assert len(splits[later]["eval_sessions"]) < len(splits[earlier]["eval_sessions"])
