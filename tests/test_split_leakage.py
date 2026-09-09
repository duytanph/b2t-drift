"""Tests that temporal splits never leak sessions or trials across time.

Run against the real T15 data files -- the whole module is skipped if
data/raw/hdf5_data_final is not present.
"""

from __future__ import annotations

import copy
import re
from datetime import date
from pathlib import Path

import pytest

from b2t_drift.data.hdf5_io import parse_session_date
from b2t_drift.data.splits import (
    build_official_split,
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


def test_train_sessions_present_in_date_order_with_zero_trial_session_recorded(sb_split) -> None:
    train_sessions = sb_split["train_sessions"]
    assert len(train_sessions) == sb_split["cut_session_index"]

    dates = [date.fromisoformat(s["session_date"]) for s in train_sessions]
    assert dates == sorted(dates)

    first = train_sessions[0]
    assert first["session_id"] == "t15.2023.08.11"
    assert first["n_trials"] == 0
    assert first["n_val_trials"] == 0


def test_t15_2023_08_11_has_zero_trials_in_every_switchboard_cut_point() -> None:
    for cut in (5, 10, 15, 20):
        split = build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=cut, corpus_filter="Switchboard")
        first = split["train_sessions"][0]
        assert first["session_id"] == "t15.2023.08.11"
        assert first["n_trials"] == 0


def test_small_training_session_still_gets_at_least_one_val_trial() -> None:
    """Uses an artificially tiny val_fraction against real (non-tiny)
    sessions so that round(val_fraction * n) == 0 for every session,
    directly exercising the max(1, ...) fix at the exact boundary it's
    meant to cover -- no session in the real data is naturally small
    enough on its own to trigger this without shrinking val_fraction."""
    split = build_temporal_split(
        DATA_ROOT, CORPUS_CSV, cut_session_index=15, corpus_filter="Switchboard", val_fraction=0.001,
    )
    val_sessions = {r[0] for r in split["val"]}
    for session in split["train_sessions"]:
        if session["n_trials"] > 0:
            assert session["n_val_trials"] == 1, session
            assert session["session_id"] in val_sessions, (
                f"{session['session_id']} has {session['n_trials']} trials "
                f"but does not appear in val"
            )


def test_validate_split_raises_on_session_moved_into_eval(sb_split) -> None:
    corrupted = copy.deepcopy(sb_split)
    # Move one training session's trials into a fabricated eval entry.
    # Pick a session with n_trials > 0 so the moved entry itself doesn't
    # trip the (separately-tested) zero-trials check instead of the
    # overlap check this test targets.
    moved_session_id = next(s["session_id"] for s in corrupted["train_sessions"] if s["n_trials"] > 0)
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

    # allow_empty_eval_sessions=True: sb_split already legitimately
    # contains real zero-trial eval sessions (see the emptiness-check
    # tests below) -- without this flag validate_split would raise on
    # one of THOSE first, which would make this test pass without
    # actually exercising the overlap check it's meant to test.
    with pytest.raises(ValueError, match=re.escape(moved_session_id)):
        validate_split(corrupted, allow_empty_eval_sessions=True)


def test_validate_split_raises_when_eval_session_precedes_cut_date(sb_split) -> None:
    corrupted = copy.deepcopy(sb_split)
    # Pick an eval session with n_trials > 0 for the same reason as above.
    target = next(e for e in corrupted["eval_sessions"] if e["n_trials"] > 0)
    target["session_date"] = corrupted["cut_date"]  # == cut_date, not after

    with pytest.raises(ValueError, match=re.escape(target["session_id"])):
        validate_split(corrupted, allow_empty_eval_sessions=True)


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
    corpus_labels = load_corpus_labels(CORPUS_CSV)
    for cut in (5, 10, 15, 20):
        split = build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=cut, corpus_filter="Switchboard")
        # allow_empty_eval_sessions=True: real Switchboard eval sessions
        # legitimately include some with n_trials==0 (docs/DESIGN.md
        # corpus stratification decision) -- this is the same flag used
        # at build time in scripts/01_build_splits.py, not a relaxation
        # specific to this test.
        validate_split(
            split, corpus_labels=corpus_labels, data_root=DATA_ROOT, allow_empty_eval_sessions=True,
        )  # must not raise


def test_later_cut_points_have_more_train_trials_and_fewer_eval_sessions() -> None:
    splits = {
        cut: build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=cut, corpus_filter="Switchboard")
        for cut in (5, 10, 15, 20)
    }
    cuts = sorted(splits)
    for earlier, later in zip(cuts, cuts[1:]):
        assert len(splits[later]["train"]) > len(splits[earlier]["train"])
        assert len(splits[later]["eval_sessions"]) < len(splits[earlier]["eval_sessions"])


# --- (b) corpus consistency check ------------------------------------------


def test_validate_split_accepts_clean_switchboard_split_with_corpus_labels(sb_split) -> None:
    corpus_labels = load_corpus_labels(CORPUS_CSV)
    validate_split(
        sb_split, corpus_labels=corpus_labels, data_root=DATA_ROOT, allow_empty_eval_sessions=True,
    )  # must not raise


def test_validate_split_raises_when_data_root_missing_but_corpus_labels_given(sb_split) -> None:
    corpus_labels = load_corpus_labels(CORPUS_CSV)
    with pytest.raises(ValueError, match="data_root"):
        validate_split(sb_split, corpus_labels=corpus_labels, allow_empty_eval_sessions=True)


def test_validate_split_raises_on_wrong_corpus_trial_injected_into_train(sb_split) -> None:
    corpus_labels = load_corpus_labels(CORPUS_CSV)
    corrupted = copy.deepcopy(sb_split)

    # t15.2023.08.11 is 100% 50-Word (see the zero-trials tests above), so
    # any of its trials belong to a different corpus than sb_split's
    # "Switchboard" filter. Borrow one from its val partition (itself a
    # 50-Word-filtered split, cut_session_index=1 keeps only session 1 as
    # train) to inject a genuinely wrong-corpus reference.
    contaminant_split = build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=1, corpus_filter="50-Word")
    assert contaminant_split["train"], "expected t15.2023.08.11 to have 50-Word trials"
    contaminant = contaminant_split["train"][0]

    corrupted["train"].append(contaminant)

    # Without corpus_labels the check doesn't run at all -- it's opt-in.
    validate_split(corrupted, allow_empty_eval_sessions=True)  # must not raise

    # With corpus_labels supplied, the contaminant is caught.
    with pytest.raises(ValueError, match=re.escape(contaminant[0])):
        validate_split(
            corrupted, corpus_labels=corpus_labels, data_root=DATA_ROOT, allow_empty_eval_sessions=True,
        )


def test_validate_split_raises_on_wrong_corpus_trial_injected_into_eval(sb_split) -> None:
    corpus_labels = load_corpus_labels(CORPUS_CSV)
    corrupted = copy.deepcopy(sb_split)

    contaminant_split = build_temporal_split(DATA_ROOT, CORPUS_CSV, cut_session_index=1, corpus_filter="50-Word")
    contaminant = contaminant_split["train"][0]

    target = next(e for e in corrupted["eval_sessions"] if e["n_trials"] > 0)
    target["trials"].append(contaminant)
    target["n_trials"] += 1

    with pytest.raises(ValueError, match=re.escape(contaminant[0])):
        validate_split(
            corrupted, corpus_labels=corpus_labels, data_root=DATA_ROOT, allow_empty_eval_sessions=True,
        )


# --- (c) emptiness checks ---------------------------------------------------


def test_validate_split_raises_on_empty_train(sb_split) -> None:
    corrupted = copy.deepcopy(sb_split)
    corrupted["train"] = []
    with pytest.raises(ValueError, match="train is empty"):
        validate_split(corrupted, allow_empty_eval_sessions=True)


def test_validate_split_raises_on_empty_val(sb_split) -> None:
    corrupted = copy.deepcopy(sb_split)
    corrupted["val"] = []
    with pytest.raises(ValueError, match="val is empty"):
        validate_split(corrupted, allow_empty_eval_sessions=True)


def test_validate_split_raises_on_empty_eval_sessions(sb_split) -> None:
    corrupted = copy.deepcopy(sb_split)
    corrupted["eval_sessions"] = []
    with pytest.raises(ValueError, match="eval_sessions is empty"):
        validate_split(corrupted, allow_empty_eval_sessions=True)


def test_validate_split_raises_by_default_on_zero_trial_eval_session(sb_split) -> None:
    # sb_split legitimately contains real zero-trial eval sessions under
    # corpus filtering (docs/DESIGN.md, corpus stratification decision) --
    # this test exercises that exact, already-present condition rather
    # than fabricating one.
    zero_trial_sessions = [e for e in sb_split["eval_sessions"] if e["n_trials"] == 0]
    assert zero_trial_sessions, "expected sb_split to contain a real zero-trial eval session"

    with pytest.raises(ValueError, match=re.escape(zero_trial_sessions[0]["session_id"])):
        validate_split(sb_split)  # allow_empty_eval_sessions defaults to False


def test_validate_split_allows_zero_trial_eval_session_when_opted_in(sb_split) -> None:
    validate_split(sb_split, allow_empty_eval_sessions=True)  # must not raise


# --- (d) CSV column validation ----------------------------------------------


def test_load_corpus_labels_raises_on_missing_column(tmp_path) -> None:
    bad_csv = tmp_path / "bad_corpus.csv"
    # "Corpus" renamed to "CorpusType" -- a plausible real-world header
    # drift, not a wildly malformed file.
    bad_csv.write_text("Date,Block number,CorpusType\n2023-08-11,1,50-Word\n")

    with pytest.raises(ValueError, match="Corpus") as exc_info:
        load_corpus_labels(bad_csv)
    message = str(exc_info.value)
    assert "Date" in message
    assert "Block number" in message
    assert "CorpusType" in message  # actual columns are named in the error too


def test_load_corpus_labels_accepts_well_formed_csv(tmp_path) -> None:
    good_csv = tmp_path / "good_corpus.csv"
    good_csv.write_text("Date,Block number,Corpus\n2023-08-11,1,50-Word\n2023-08-11,2,Switchboard\n")

    labels = load_corpus_labels(good_csv)
    assert labels[("t15.2023.08.11", 1)] == "50-Word"
    assert labels[("t15.2023.08.11", 2)] == "Switchboard"


# --- (e) official split (Model A pipeline check) ----------------------------


@pytest.fixture(scope="module")
def official_split():
    return build_official_split(DATA_ROOT)


def test_official_split_covers_all_45_sessions(official_split) -> None:
    session_to_index = official_split["session_to_index"]
    assert len(session_to_index) == 45
    assert sorted(session_to_index.values()) == list(range(45))


def test_official_split_train_val_counts_match_published_inventory(official_split) -> None:
    # docs/DATASET.md section 11's verified inventory: 8,072 train / 1,426 val.
    assert len(official_split["train"]) == 8072
    assert len(official_split["val"]) == 1426


def test_official_split_has_no_eval_sessions_and_all_corpora(official_split) -> None:
    assert official_split["eval_sessions"] == []
    assert official_split["corpus_filter"] is None
    assert official_split["kind"] == "official"


def test_official_split_validates_cleanly(official_split) -> None:
    validate_split(official_split)  # must not raise


def test_official_split_train_val_disjoint_per_session(official_split) -> None:
    # Official train/val come from separate HDF5 files (data_train vs.
    # data_val), so no trial reference should appear in both.
    train_refs = {tuple(r) for r in official_split["train"]}
    val_refs = {tuple(r) for r in official_split["val"]}
    assert train_refs.isdisjoint(val_refs)


def test_validate_split_rejects_official_manifest_with_nonempty_eval_sessions(official_split) -> None:
    bad = copy.deepcopy(official_split)
    bad["eval_sessions"] = [{"session_id": "t15.2023.10.20"}]
    with pytest.raises(ValueError, match="official"):
        validate_split(bad)


def test_validate_split_defaults_missing_kind_to_temporal(sb_split) -> None:
    # Manifests written before the "kind" field existed have no such key --
    # validate_split must still apply the temporal (cut-date) checks to them.
    no_kind = copy.deepcopy(sb_split)
    del no_kind["kind"]
    # allow_empty_eval_sessions=True: sb_split (Switchboard s15) genuinely
    # has a zero-trial eval session under corpus filtering, same as
    # 01_build_splits.py's real call -- unrelated to the kind-default
    # behavior this test targets.
    validate_split(no_kind, allow_empty_eval_sessions=True)  # must not raise -- defaults to "temporal"


def test_validate_split_rejects_unknown_kind(sb_split) -> None:
    bad = copy.deepcopy(sb_split)
    bad["kind"] = "bogus"
    with pytest.raises(ValueError, match="bogus"):
        validate_split(bad)
