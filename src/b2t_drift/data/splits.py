"""Temporal session splits: building, (de)serializing, and validating them.

WHY VERSIONED ARTIFACTS, NOT COMPUTED INLINE. A split assignment is a
scientific decision (which sessions are "calibration," which are "the
future") that must be identical every time it's used -- in training, in
evaluation, and in anyone re-checking the result later. Recomputing it
inline in a training script risks two runs silently using different splits
(a config default changes, a session gets added to the data directory,
someone edits a cut point) with no record of what actually happened.
Writing it once to `splits/*.json` (`docs/CLAUDE.md`) makes the split
itself an auditable, diffable, git-tracked fact, not a side effect of
whatever code happened to run.

THE LEAKAGE RISK, AND WHY VALIDATION HERE IS AGGRESSIVE. This project's
entire result is one claim: every training trial's session date precedes
every evaluation trial's session date. If that claim is false for even one
trial, the drift curve measures something else -- interpolation, not
extrapolation -- and every PER number downstream is wrong in a way that
does NOT show up as an error. A leakage bug produces a model that trains,
converges, and reports a perfectly plausible-looking (and *better than it
should be*) PER curve; nothing crashes, nothing looks obviously off, and
the mistake is only visible to someone who goes back and checks session
dates by hand. `validate_split` exists to make that check automatic,
repeated on every manifest this module writes, and specific enough
(naming the offending session or trial, not just "leakage detected") that
a failure is actually actionable.

THE CORPUS-LABEL-VS-COUNT RULE. `t15_copyTaskData_description.csv` is
authoritative for CORPUS LABELS (which corpus a block belongs to) but NOT
for trial COUNTS -- its per-block sentence counts have been found to
disagree with the actual released HDF5 files for a nontrivial fraction of
blocks, and do not distinguish labeled from unlabeled partitions at all
(`docs/FINDINGS.md`: "Corpus CSV trial counts are unreliable; the manifest
is authoritative"). Trial counts always come
from `hdf5_io` (attribute reads against the real files) or the
`session_manifest.csv` it produces. `load_corpus_labels` below returns
labels only, by design, and its docstring says so explicitly so this rule
doesn't have to be rediscovered.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import subprocess
import warnings
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from b2t_drift.data.hdf5_io import (
    TrialId,
    discover_sessions,
    iter_trial_ids,
    parse_session_date,
)

DEFAULT_VAL_FRACTION = 0.1
DEFAULT_MIN_EVAL_TRIALS = 50
CREATED_BY = "scripts/01_build_splits.py"


def load_corpus_labels(csv_path: Path) -> dict[tuple[str, int], str]:
    """Load block-to-corpus labels from the copy-task description CSV.

    LABELS ONLY -- see the module docstring's "corpus-label-vs-count rule."
    Never use this function, or this CSV, for trial counts; get those from
    `hdf5_io` / `session_manifest.csv` instead.

    Args:
        csv_path: Path to `t15_copyTaskData_description.csv`.

    Returns:
        A dict mapping `(session_id, block_num)` to the corpus name for
        that block, e.g. `("t15.2023.08.13", 1) -> "Switchboard"`.
        `session_id` is derived from the CSV's `Date` column (`YYYY-MM-DD`
        -> `t15.YYYY.MM.DD`), not read from any HDF5 file.
    """
    required_columns = ("Date", "Block number", "Corpus")
    labels: dict[tuple[str, int], str] = {}
    with Path(csv_path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        actual_columns = reader.fieldnames or []
        missing = [c for c in required_columns if c not in actual_columns]
        if missing:
            raise ValueError(
                f"load_corpus_labels: {csv_path} is missing expected "
                f"column(s) {missing}. Expected: {list(required_columns)}. "
                f"Actual: {list(actual_columns)}."
            )
        for row in reader:
            session_id = "t15." + row["Date"].replace("-", ".")
            block_num = int(row["Block number"])
            labels[(session_id, block_num)] = row["Corpus"]
    return labels


def _git_commit_hash() -> str | None:
    """Best-effort current commit hash, or None if unavailable.

    Returns:
        The full commit SHA `git rev-parse HEAD` reports, or `None` if
        `git` isn't installed, this isn't a git repository, or the call
        fails for any other reason -- a manifest must still be buildable
        outside a git checkout.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _trials_for_session(
    data_root: Path,
    session_id: str,
    corpus_labels: dict[tuple[str, int], str] | None,
    corpus_filter: str | None,
) -> list[TrialId]:
    """Labeled (train+val) trials for one session, optionally corpus-filtered.

    Args:
        data_root: Directory containing per-session subdirectories.
        session_id: Session to enumerate.
        corpus_labels: Result of `load_corpus_labels`, or None if
            `corpus_filter` is None (no filtering needed).
        corpus_filter: If given, keep only trials whose block is labeled
            with this corpus.

    Returns:
        `TrialId`s for this session's train+val trials, in `iter_trial_ids`
        order, filtered by corpus if requested.

    Raises:
        ValueError: If `corpus_filter` is given and a trial's
            `(session_id, block_num)` has no entry in `corpus_labels` --
            every block was found to have a corpus label when this CSV was
            inspected (`docs/FINDINGS.md`: "Corpus CSV trial counts are
            unreliable"), so a missing lookup
            here means the data has changed and should not be silently
            skipped.
    """
    trials = []
    for trial_id in iter_trial_ids(data_root, session_id, partitions=("train", "val")):
        if corpus_filter is not None:
            key = (trial_id.session_id, trial_id.block_num)
            corpus = corpus_labels.get(key) if corpus_labels else None
            if corpus is None:
                raise ValueError(
                    f"_trials_for_session: no corpus label found for "
                    f"{key} -- cannot apply corpus_filter={corpus_filter!r}."
                )
            if corpus != corpus_filter:
                continue
        trials.append(trial_id)
    return trials


def build_temporal_split(
    data_root: Path,
    corpus_csv: Path,
    cut_session_index: int,
    corpus_filter: str | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    min_eval_trials: int = DEFAULT_MIN_EVAL_TRIALS,
) -> dict[str, Any]:
    """Build one temporal train/val/eval split manifest.

    Sessions are ordered BY DATE (`hdf5_io.discover_sessions`), never by
    directory listing order. The first `cut_session_index` sessions (by
    date) become the training block; `cut_date` is the last training
    session's date. Every later session becomes one `eval_sessions` entry
    -- including sessions with zero or few matching trials after corpus
    filtering, which are FLAGGED (`passes_threshold: false`) rather than
    silently dropped, so the manifest itself is auditable against the
    ≥`min_eval_trials` threshold.

    The validation set is sampled from WITHIN the training sessions only,
    stratified by session so no training session with at least 1 trial is
    entirely absent from validation, and no validation trial's session
    postdates the cut: each training session with `n >= 1` trials
    contributes `max(1, round(val_fraction * n))` trials (capped at `n`),
    chosen by `random.Random(seed).sample`. The `max(1, ...)` matters --
    `round(val_fraction * n)` alone returns 0 for any session with fewer
    than `1 / (2 * val_fraction)` trials (5 trials at the default
    `val_fraction=0.1`), which would silently exclude small sessions from
    validation entirely despite the "stratified by session" claim.

    A TRAINING SESSION CAN LEGITIMATELY HAVE 0 TRIALS after corpus
    filtering (e.g. `t15.2023.08.11`, the earliest session, is 100%
    50-Word -- 0 of its 288 trials are Switchboard). This is NOT treated
    as an error: the session contributes 0 trials to both `train` and
    `val`, a `UserWarning` is emitted naming it, and it is recorded in the
    returned `train_sessions` list with `n_trials: 0` -- an auditable fact
    in the committed manifest, not a silent skip. (An earlier version of
    this function raised `ValueError` here; that was wrong, because
    `t15.2023.08.11` is the first session in every one of this project's
    cut points, so it would have made every Switchboard-only manifest
    unbuildable. `validate_split` still raises if the training block ends
    up EMPTY, i.e. every training session had 0 trials -- see its
    docstring.)

    Args:
        data_root: Directory containing per-session subdirectories.
        corpus_csv: Path to `t15_copyTaskData_description.csv`, used only
            if `corpus_filter` is given (see `load_corpus_labels`).
        cut_session_index: Number of earliest-by-date sessions to put in
            the training block (e.g. `15` for sessions 1-15).
        corpus_filter: If given (e.g. `"Switchboard"`), keep only trials
            from blocks labeled with this corpus, in BOTH the training
            block and every eval session -- matching train/eval corpus
            distributions is the point (`docs/DESIGN.md`, Switchboard-only
            decision). If None, all corpora are included.
        val_fraction: Fraction of each training session's trials to hold
            out for validation.
        seed: Seed for the validation-trial sampling.
        min_eval_trials: Threshold an eval session's trial count must meet
            to be marked `passes_threshold: true`. Should be fixed before
            looking at any PER value (`docs/DESIGN.md` section 7).

    Returns:
        A dict matching the schema documented in `scripts/01_build_splits.py`
        / the project's manifest format: `name`, `created_by`, `created_at`,
        `git_commit`, `corpus_filter`, `cut_session_index`, `cut_date`,
        `val_fraction`, `seed`, `session_to_index`, `train_sessions`,
        `train`, `val`, `eval_sessions`. Trial references are
        `[session_id, partition, trial_key]` lists (JSON-array-shaped, not
        tuples, so a `write_split` -> `load_split` round trip compares
        equal). `train_sessions` is a list of `{session_id, session_date,
        n_trials, n_val_trials}` dicts, one per training-block session in
        date order, with counts AFTER corpus filtering -- this is what
        makes a 0-trial training session (see above) an explicit,
        auditable entry rather than an absence someone has to notice on
        their own. `session_to_index` covers ALL sessions in `data_root`
        (not just ones with trials surviving this split's corpus filter),
        assigned by date order -- the same mapping regardless of
        `corpus_filter` or `cut_session_index`, so session indices are
        stable across every manifest this function produces (needed for
        the Week 3 day-specific input layers to refer to "the same"
        session consistently across cut points).

    Raises:
        ValueError: If `cut_session_index` is out of range for the number
            of discovered sessions, or if (via `_trials_for_session`)
            `corpus_filter` is given and some block has no corpus label.
    """
    sessions = discover_sessions(data_root)
    if not (1 <= cut_session_index < len(sessions)):
        raise ValueError(
            f"build_temporal_split: cut_session_index={cut_session_index} "
            f"out of range for {len(sessions)} discovered sessions "
            f"(must leave at least 1 session for eval)."
        )

    session_to_index = {s.session_id: i for i, s in enumerate(sessions)}

    corpus_labels = load_corpus_labels(corpus_csv) if corpus_filter is not None else None

    train_session_files = sessions[:cut_session_index]
    eval_sessions_meta = sessions[cut_session_index:]
    cut_date = train_session_files[-1].session_date

    rng = random.Random(seed)
    train_refs: list[list[str]] = []
    val_refs: list[list[str]] = []
    train_sessions: list[dict[str, Any]] = []
    for session in train_session_files:
        trials = _trials_for_session(data_root, session.session_id, corpus_labels, corpus_filter)
        n_trials = len(trials)
        if n_trials == 0:
            n_val = 0
            warnings.warn(
                f"build_temporal_split: training session {session.session_id!r} "
                f"has 0 trials after corpus filtering (corpus_filter="
                f"{corpus_filter!r}) -- it contributes nothing to training. "
                f"Recorded in the manifest's train_sessions list with "
                f"n_trials=0, not silently dropped.",
                stacklevel=2,
            )
        else:
            n_val = min(n_trials, max(1, round(val_fraction * n_trials)))
        val_indices = set(rng.sample(range(n_trials), n_val)) if n_val else set()
        for i, trial_id in enumerate(trials):
            ref = [trial_id.session_id, trial_id.partition, trial_id.trial_key]
            (val_refs if i in val_indices else train_refs).append(ref)
        train_sessions.append({
            "session_id": session.session_id,
            "session_date": session.session_date.isoformat(),
            "n_trials": n_trials,
            "n_val_trials": n_val,
        })

    eval_sessions: list[dict[str, Any]] = []
    for session in eval_sessions_meta:
        trials = _trials_for_session(data_root, session.session_id, corpus_labels, corpus_filter)
        seq_lens = [t.seq_len for t in trials]
        eval_sessions.append({
            "session_id": session.session_id,
            "session_date": session.session_date.isoformat(),
            "days_since_cut": (session.session_date - cut_date).days,
            "n_trials": len(trials),
            "median_seq_len": float(statistics.median(seq_lens)) if seq_lens else None,
            "passes_threshold": len(trials) >= min_eval_trials,
            "trials": [[t.session_id, t.partition, t.trial_key] for t in trials],
        })

    corpus_slug = corpus_filter.lower() if corpus_filter is not None else "allcorpora"
    name = f"temporal_s01_s{cut_session_index:02d}_{corpus_slug}"

    return {
        "name": name,
        "created_by": CREATED_BY,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "corpus_filter": corpus_filter,
        "cut_session_index": cut_session_index,
        "cut_date": cut_date.isoformat(),
        "val_fraction": val_fraction,
        "seed": seed,
        "session_to_index": session_to_index,
        "train_sessions": train_sessions,
        "train": train_refs,
        "val": val_refs,
        "eval_sessions": eval_sessions,
    }


def write_split(split_dict: dict[str, Any], path: Path) -> None:
    """Write a split manifest to a JSON file, creating parent directories.

    Args:
        split_dict: A manifest, as returned by `build_temporal_split`.
        path: Destination path, conventionally `splits/{name}.json`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(split_dict, f, indent=2)


def load_split(path: Path) -> dict[str, Any]:
    """Load a previously written split manifest.

    Args:
        path: Path to a JSON manifest under `splits/`.

    Returns:
        The manifest dict, structurally identical to what `write_split`
        was given (trial references come back as lists, matching how
        `build_temporal_split` constructs them, so `write_split` then
        `load_split` round-trips to an equal dict).
    """
    with path.open("r") as f:
        return json.load(f)


def _eval_session_trials(split_dict: dict[str, Any], session_id: str) -> list[list[str]]:
    """Trial references for one named eval session.

    Raises:
        ValueError: If `session_id` is not one of `split_dict`'s
            `eval_sessions`.
    """
    for entry in split_dict["eval_sessions"]:
        if entry["session_id"] == session_id:
            return entry["trials"]
    raise ValueError(
        f"_eval_session_trials: {session_id!r} is not an eval session in "
        f"this manifest."
    )


def _resolve_trial_ids(
    refs: list[list[str]],
    data_root: Path,
) -> dict[tuple[str, str, str], TrialId]:
    """Look up full `TrialId`s for a list of trial references.

    Shared by `split_to_trial_ids` and `validate_split`'s (opt-in) corpus
    consistency check. Reads each distinct `(session_id, partition)` file's
    attributes exactly once via `iter_trial_ids`, regardless of how many
    of `refs` come from it.

    Args:
        refs: `[session_id, partition, trial_key]` references.
        data_root: Directory containing per-session subdirectories.

    Returns:
        A dict from `(session_id, partition, trial_key)` to the matching
        `TrialId`, for every reference that was found. References not
        found in `data_root` are simply absent from the result -- callers
        that need to treat that as an error check for missing keys
        themselves (see `split_to_trial_ids`).
    """
    needed_keys: dict[tuple[str, str], set[str]] = {}
    for session_id, part, trial_key in refs:
        needed_keys.setdefault((session_id, part), set()).add(trial_key)

    lookup: dict[tuple[str, str, str], TrialId] = {}
    for (session_id, part), wanted in needed_keys.items():
        for trial_id in iter_trial_ids(data_root, session_id, partitions=(part,)):
            if trial_id.trial_key in wanted:
                lookup[(session_id, part, trial_id.trial_key)] = trial_id
    return lookup


def split_to_trial_ids(
    split_dict: dict[str, Any],
    data_root: Path,
    partition: str,
) -> list[TrialId]:
    """Rehydrate a manifest partition's trial references into `TrialId`s.

    Trial references only store `[session_id, partition, trial_key]`
    (`docs/DESIGN.md` / the manifest schema) -- everything else
    (`n_time_steps`, `seq_len`, `sentence_label`, ...) is re-read from the
    HDF5 attributes here, once per distinct `(session_id, partition)` file
    touched (via `_resolve_trial_ids` / `iter_trial_ids`), rather than
    duplicated into the JSON.

    Args:
        split_dict: A manifest, as returned by `build_temporal_split` or
            `load_split`.
        data_root: Directory containing per-session subdirectories.
        partition: `"train"`, `"val"`, or an eval session's `session_id`
            (e.g. `"t15.2023.10.20"`) to rehydrate that eval session's
            trials specifically.

    Returns:
        `TrialId`s in the same order as the manifest's trial references
        for this partition.

    Raises:
        ValueError: If `partition` is not `"train"`, `"val"`, or a known
            eval `session_id`; or if a trial reference in the manifest is
            not found when re-reading `data_root` (the data may have
            changed since this manifest was built).
    """
    if partition == "train":
        refs = split_dict["train"]
    elif partition == "val":
        refs = split_dict["val"]
    else:
        refs = _eval_session_trials(split_dict, partition)

    lookup = _resolve_trial_ids(refs, data_root)

    result = []
    for session_id, part, trial_key in refs:
        key = (session_id, part, trial_key)
        if key not in lookup:
            raise ValueError(
                f"split_to_trial_ids: trial reference {key} from the "
                f"manifest was not found under {data_root} -- the data may "
                f"have changed since this manifest was built."
            )
        result.append(lookup[key])
    return result


def validate_split(
    split_dict: dict[str, Any],
    corpus_labels: dict[tuple[str, int], str] | None = None,
    data_root: Path | None = None,
    allow_empty_eval_sessions: bool = False,
) -> None:
    """Aggressively check a manifest for temporal or structural leakage.

    Recomputes every property from the manifest's raw trial references and
    session lists -- it does not trust that whatever built the manifest
    got the bookkeeping right, because a bug that gets this wrong produces
    a manifest that looks completely normal (see the module docstring).

    Args:
        split_dict: A manifest, as returned by `build_temporal_split` or
            `load_split`.
        corpus_labels: Result of `load_corpus_labels`. If given (and
            `split_dict["corpus_filter"]` is not None), additionally
            checks that EVERY trial reference in train, val, and every
            eval session actually belongs to that corpus. This check
            reads HDF5 attributes (to resolve each reference's
            `block_num`, which trial references don't carry -- see
            `_resolve_trial_ids`) and is therefore much slower than every
            other check here, which is why it's opt-in rather than
            always-on: `data_root` must also be given when `corpus_labels`
            is, since resolving `block_num` requires reading the actual
            files.
        data_root: Directory containing per-session subdirectories.
            Required if `corpus_labels` is given; ignored otherwise.
        allow_empty_eval_sessions: If False (the default), an eval session
            with `n_trials == 0` raises. Such a session is a LEGITIMATE
            outcome under corpus filtering (a held-out session can
            genuinely contain zero trials of the target corpus), and the
            manifest is designed to keep it as a flagged entry rather than
            drop it (`passes_threshold: false` -- `docs/DESIGN.md`,
            corpus stratification decision). But whether that's
            acceptable for a given use of the manifest is a decision, not
            a default: the strict default forces it to be made
            explicitly, by passing `allow_empty_eval_sessions=True`,
            rather than silently proceeding with a curve point that has
            nothing behind it.

    Raises:
        ValueError: On the first of the following that fails, naming the
            specific offending session or trial rather than just stating
            that a problem was found:
              - `train`, `val`, or `eval_sessions` is empty
              - an eval session has `n_trials == 0` (unless
                `allow_empty_eval_sessions=True`)
              - a session appears in both train/val and `eval_sessions`
              - any eval session's date is on or before `cut_date`
              - any training/val session's date is after `cut_date`
              - a trial reference appears in more than one of
                {train, val, any eval session}
              - `session_to_index` is missing a session referenced
                anywhere in the manifest
              - `session_to_index`'s values are not contiguous from 0
              - an eval session's `days_since_cut` does not equal
                `session_date - cut_date`
              - (only when `corpus_labels` is given) a trial reference
                does not belong to `split_dict["corpus_filter"]`
    """
    cut_date = date.fromisoformat(split_dict["cut_date"])
    train_refs = split_dict["train"]
    val_refs = split_dict["val"]
    eval_sessions = split_dict["eval_sessions"]
    session_to_index = split_dict["session_to_index"]

    if not train_refs:
        raise ValueError("validate_split: train is empty.")
    if not val_refs:
        raise ValueError("validate_split: val is empty.")
    if not eval_sessions:
        raise ValueError("validate_split: eval_sessions is empty.")

    if not allow_empty_eval_sessions:
        for e in eval_sessions:
            if e["n_trials"] == 0:
                raise ValueError(
                    f"validate_split: eval session {e['session_id']!r} has "
                    f"n_trials=0. This can be a legitimate outcome under "
                    f"corpus filtering, but must be explicitly accepted -- "
                    f"pass allow_empty_eval_sessions=True if so."
                )

    train_val_session_ids = {ref[0] for ref in train_refs} | {ref[0] for ref in val_refs}
    eval_session_ids = {e["session_id"] for e in eval_sessions}

    overlap = train_val_session_ids & eval_session_ids
    if overlap:
        raise ValueError(
            f"validate_split: session(s) {sorted(overlap)} appear in both "
            f"train/val and eval_sessions."
        )

    for e in eval_sessions:
        eval_date = date.fromisoformat(e["session_date"])
        if eval_date <= cut_date:
            raise ValueError(
                f"validate_split: eval session {e['session_id']!r} has date "
                f"{eval_date.isoformat()}, which is on or before "
                f"cut_date={cut_date.isoformat()}."
            )
        expected_offset = (eval_date - cut_date).days
        if e["days_since_cut"] != expected_offset:
            raise ValueError(
                f"validate_split: eval session {e['session_id']!r} has "
                f"days_since_cut={e['days_since_cut']}, but session_date "
                f"{eval_date.isoformat()} minus cut_date "
                f"{cut_date.isoformat()} is {expected_offset}."
            )

    for session_id in sorted(train_val_session_ids):
        session_date = parse_session_date(session_id)
        if session_date > cut_date:
            raise ValueError(
                f"validate_split: training/val session {session_id!r} has "
                f"date {session_date.isoformat()}, which is after "
                f"cut_date={cut_date.isoformat()}."
            )

    group_of: dict[tuple[str, str, str], list[str]] = {}
    for ref in train_refs:
        group_of.setdefault(tuple(ref), []).append("train")
    for ref in val_refs:
        group_of.setdefault(tuple(ref), []).append("val")
    for e in eval_sessions:
        for ref in e["trials"]:
            group_of.setdefault(tuple(ref), []).append(f"eval:{e['session_id']}")
    leaked = {ref: groups for ref, groups in group_of.items() if len(groups) > 1}
    if leaked:
        first_ref, first_groups = next(iter(leaked.items()))
        raise ValueError(
            f"validate_split: trial reference {list(first_ref)} appears in "
            f"more than one partition: {first_groups} "
            f"({len(leaked)} total trial(s) affected)."
        )

    referenced_sessions = set(train_val_session_ids) | set(eval_session_ids)
    missing = referenced_sessions - set(session_to_index)
    if missing:
        raise ValueError(
            f"validate_split: session_to_index is missing session(s) "
            f"{sorted(missing)}."
        )

    index_values = sorted(session_to_index.values())
    if index_values != list(range(len(index_values))):
        raise ValueError(
            f"validate_split: session_to_index values are not contiguous "
            f"from 0: {index_values}."
        )

    if corpus_labels is not None:
        if data_root is None:
            raise ValueError(
                "validate_split: data_root is required when corpus_labels "
                "is given -- the corpus consistency check resolves each "
                "trial reference's block_num by reading HDF5 attributes."
            )
        expected_corpus = split_dict["corpus_filter"]
        if expected_corpus is not None:
            tagged_refs: list[tuple[str, list[str]]] = [("train", r) for r in train_refs]
            tagged_refs += [("val", r) for r in val_refs]
            for e in eval_sessions:
                tagged_refs += [(f"eval:{e['session_id']}", r) for r in e["trials"]]

            lookup = _resolve_trial_ids([r for _, r in tagged_refs], data_root)
            for group_name, ref in tagged_refs:
                key = tuple(ref)
                trial_id = lookup.get(key)
                if trial_id is None:
                    raise ValueError(
                        f"validate_split: trial reference {ref} (in "
                        f"{group_name}) was not found under {data_root}."
                    )
                actual_corpus = corpus_labels.get((trial_id.session_id, trial_id.block_num))
                if actual_corpus != expected_corpus:
                    raise ValueError(
                        f"validate_split: trial reference {ref} (in "
                        f"{group_name}) belongs to corpus "
                        f"{actual_corpus!r}, expected {expected_corpus!r} "
                        f"(split_dict['corpus_filter'])."
                    )
