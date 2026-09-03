"""Low-level HDF5 reading for the T15 dataset.

Responsible for discovering session/partition HDF5 files under
`data/raw/hdf5_data_final/` and reading trial-group data out of them. This
is the ONLY module in the codebase that knows HDF5 exists.

SCOPE. This module reads. It does not normalize, batch, pad, convert to
tensors, or split. Those responsibilities belong to `data/dataset.py`,
`data/collate.py`, and `data/splits.py` respectively -- do not add them
here, even for convenience. `TrialId` (below) exists specifically so
`splits.py` can enumerate and partition trials using only attribute-level
facts (session, date, block, length), never touching a feature array.

Confirmed on-disk layout (data/raw/hdf5_data_final/):
    <data_root>/t15.YYYY.MM.DD/data_{train,val,test}.hdf5
Every session has `data_train.hdf5`; 41 of 45 sessions also have
`data_val.hdf5` and `data_test.hdf5` (the remaining 4 have only the train
file). Each file's top-level keys are trial groups named `trial_NNNN`, each
holding:
    input_features  (n_time_steps, 512) float32   -- variable length, NOT padded
    seq_class_ids   (500,) int32                  -- padded; real length is seq_len
    transcription   (500,) int32                  -- padded
Every trial group has attributes `n_time_steps`, `session`, `block_num`,
`trial_num`. Trials also have `seq_len` and `sentence_label` attributes
when ground-truth labels are released for that partition -- observed to be
absent on every trial in the `test` partition.

`transcription` IS DELIBERATELY NEVER READ. It decodes to exactly the same
string as the `sentence_label` attribute (verified 200/200 sampled trials
-- `docs/DATASET.md` section 8), so reading it would be pure duplicated
I/O against a ~12 GB dataset for zero additional information.

FILE HANDLE MANAGEMENT AND MULTIPROCESSING. `read_trial`/`read_trials` use
a small process-local LRU cache of open `h5py.File` handles (see
`_get_file_handle`, `close_all`), capped at `_MAX_OPEN_HANDLES` and keyed
by `(os.getpid(), file_path)`.

CRITICAL: h5py file handles cannot be shared across OS processes. If a
handle opened in a parent process is inherited by a `DataLoader` worker
(`num_workers > 0`) -- e.g. because something opened a file eagerly before
the workers were forked/spawned -- reads through that inherited handle
produce corrupted data or silently hang. This class of bug often appears
to work fine on a small/serial test run and then fails only under real
multi-worker load, which is what makes it dangerous. Two properties of the
cache exist specifically to prevent this:
  - Handles are opened LAZILY, only on first real access inside
    `read_trial`/`read_trials`, never in a constructor and never at
    import time. This means handles only ever come into existence inside
    whichever process actually performs a read -- a `Dataset.__init__`
    that runs in the parent process before workers fork can never
    accidentally open (and thus hand down) a handle.
  - The cache key includes `os.getpid()`. A worker process therefore never
    looks up (and so never reuses) a handle path that was opened by a
    different process; it transparently opens its own fresh handle on
    first access instead. Call `close_all()` for explicit cleanup (e.g.
    at the end of a script or between test cases) -- it only closes
    handles belonging to the calling process's pid.
"""

from __future__ import annotations

import csv
import os
import statistics
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SESSION_DIR_PREFIX = "t15."
PARTITIONS = ("train", "val", "test")
PARTITION_FILENAMES = {
    "train": "data_train.hdf5",
    "val": "data_val.hdf5",
    "test": "data_test.hdf5",
}

# Attributes present on every trial group, regardless of partition.
REQUIRED_TRIAL_ATTRS = ("n_time_steps", "session", "block_num", "trial_num")

# Attributes only present when ground-truth labels are released for a
# partition (observed: absent for the `test` partition).
LABEL_TRIAL_ATTRS = ("seq_len", "sentence_label")

# The only partition-presence patterns observed in the data: every session
# has a train file, and either has both val+test too, or has neither.
EXPECTED_PARTITION_PATTERNS = (frozenset({"train"}), frozenset(PARTITIONS))


@dataclass(frozen=True)
class SessionFiles:
    """Which partition files exist for one session."""

    session_id: str
    session_date: date
    partitions: dict[str, Path]  # partition name -> hdf5 file path


@dataclass(frozen=True)
class TrialAttrIssue:
    """A trial group missing an attribute that was expected to be present."""

    session_id: str
    partition: str
    file_path: Path
    trial_key: str
    missing_attr: str


@dataclass(frozen=True)
class SessionManifestResult:
    """Everything `scripts/00_verify_data.py` needs to report and write."""

    rows: list[dict[str, Any]]
    issues: list[TrialAttrIssue]
    sessions: list[SessionFiles]
    unexpected_partition_patterns: list[tuple[str, frozenset[str]]]


@dataclass(frozen=True)
class TrialId:
    """Identifies one trial's location and metadata, without its arrays.

    `splits.py` must enumerate and partition trials -- by session, date,
    block, and length -- without touching any feature array (the released
    dataset is ~12 GB; building a split over the whole thing must stay an
    attribute-only operation). Keeping identity (this class) separate from
    data (`read_trial`) is what makes split construction fast: a `TrialId`
    is built entirely from HDF5 attributes via `iter_trial_ids`.

    Being `frozen` (and therefore hashable) is required, not incidental --
    a `TrialId` needs to work as a dict key and inside sets wherever split
    or caching logic needs to deduplicate or index by trial.

    Attributes:
        session_id: Session identifier, e.g. `"t15.2023.08.11"`.
        session_date: The session's recording date, parsed from
            `session_id`.
        partition: Which file this trial lives in: `"train"`, `"val"`, or
            `"test"`.
        trial_key: The trial's HDF5 group key, e.g. `"trial_0149"`. Not the
            same as `trial_num` -- see `docs/DATASET.md` section 6.
        block_num: Recording block this trial belongs to.
        trial_num: Trial index within its partition FILE (not global --
            restarts at 0 in each of train/val/test).
        n_time_steps: True (unpadded) length of `input_features` for this
            trial, i.e. `input_features.shape[0]`.
        seq_len: True (unpadded) length of `seq_class_ids` for this trial.
            For `test`-partition trials, this is `0` -- the HDF5 file has
            no `seq_len` attribute there (the test partition is
            unlabeled), and `0` is a placeholder, not an invented real
            value. Check `has_labels` rather than comparing to 0 directly,
            so the placeholder convention only needs to be known in one
            place.
        sentence_label: Ground-truth sentence text. For `test`-partition
            trials this is `""`, for the same reason as `seq_len` above.
    """

    session_id: str
    session_date: date
    partition: str
    trial_key: str
    block_num: int
    trial_num: int
    n_time_steps: int
    seq_len: int
    sentence_label: str

    @property
    def has_labels(self) -> bool:
        """Whether this trial has real (not placeholder) labels.

        Returns:
            `seq_len > 0`. False for every `test`-partition trial.
        """
        return self.seq_len > 0


def parse_session_date(session_id: str) -> date:
    """Parse a session id like `t15.2023.08.13` into a `date`.

    Args:
        session_id: Session identifier, e.g. `t15.2023.08.13`.

    Returns:
        The recording date.
    """
    _, year, month, day = session_id.split(".")
    return date(int(year), int(month), int(day))


def discover_sessions(data_root: Path) -> list[SessionFiles]:
    """Enumerate session directories and the partition files each has.

    Args:
        data_root: Directory containing per-session subdirectories
            (`t15.YYYY.MM.DD/`), each holding `data_{train,val,test}.hdf5`.

    Returns:
        One `SessionFiles` per session subdirectory, sorted by session date.
    """
    sessions = []
    for entry in data_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith(SESSION_DIR_PREFIX):
            continue
        session_id = entry.name
        partitions = {}
        for partition, filename in PARTITION_FILENAMES.items():
            candidate = entry / filename
            if candidate.exists():
                partitions[partition] = candidate
        sessions.append(
            SessionFiles(
                session_id=session_id,
                session_date=parse_session_date(session_id),
                partitions=partitions,
            )
        )
    return sorted(sessions, key=lambda s: s.session_date)


def list_session_files(data_root: Path) -> list[Path]:
    """Enumerate all partition HDF5 files under a data root.

    Args:
        data_root: Directory containing per-session subdirectories.

    Returns:
        Paths to every existing partition file, across all sessions,
        ordered by session date then by partition (train, val, test).
    """
    files = []
    for session in discover_sessions(data_root):
        for partition in PARTITIONS:
            if partition in session.partitions:
                files.append(session.partitions[partition])
    return files


def session_id_from_path(path: Path) -> str:
    """Derive a session id from a partition file path.

    Args:
        path: Path to a `data_{train,val,test}.hdf5` file, expected to sit
            directly inside its session directory.

    Returns:
        The owning session's id, e.g. `t15.2023.08.13`.
    """
    return path.parent.name


def read_session(path: Path) -> dict[str, np.ndarray]:
    """Read one partition file's full trial arrays into memory.

    TODO: not implemented, and likely never should be as written -- eagerly
    loading a whole partition file's arrays reintroduces the ~12 GB-vs-16 GB
    RAM problem `data/dataset.py`'s lazy loading exists to avoid. Per-trial
    reading is implemented instead: see `iter_trial_ids` (attributes only)
    and `read_trial` / `read_trials` (arrays for specific trials).

    Args:
        path: Path to a single `data_{train,val,test}.hdf5` file.

    Returns:
        A dict of raw arrays (neural features, phoneme labels, and any
        associated metadata) per trial.
    """
    raise NotImplementedError


def iter_trial_attributes(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (trial_key, attrs) for every trial group in an HDF5 file.

    Reads only HDF5 attributes -- never touches the `input_features`,
    `seq_class_ids`, or `transcription` datasets. This keeps the operation
    fast even though the underlying files are large.

    Args:
        path: Path to a `data_{train,val,test}.hdf5` file.

    Yields:
        `(trial_key, attrs)` pairs in file iteration order, where `attrs`
        is a plain dict copy of the trial group's HDF5 attributes.
    """
    with h5py.File(path, "r") as f:
        for trial_key in f.keys():
            yield trial_key, dict(f[trial_key].attrs)


def summarize_partition(
    session_id: str,
    session_date: date,
    partition: str,
    path: Path,
) -> tuple[dict[str, Any], list[TrialAttrIssue]]:
    """Summarize one (session, partition) HDF5 file from trial attributes only.

    Args:
        session_id: Owning session id, e.g. `t15.2023.08.13`.
        session_date: Owning session's recording date.
        partition: One of `train`, `val`, `test`.
        path: Path to the partition's HDF5 file.

    Returns:
        A tuple of (manifest row dict matching the
        `data/interim/session_manifest.csv` schema, list of
        `TrialAttrIssue`s for trials missing an attribute that was expected
        to be present).
    """
    n_trials = 0
    block_nums: set[int] = set()
    time_bins: list[int] = []
    seq_lens: list[int] = []
    issues: list[TrialAttrIssue] = []
    expects_labels = partition != "test"

    for trial_key, attrs in iter_trial_attributes(path):
        n_trials += 1

        for attr_name in REQUIRED_TRIAL_ATTRS:
            if attr_name not in attrs:
                issues.append(
                    TrialAttrIssue(session_id, partition, path, trial_key, attr_name)
                )
        if expects_labels:
            for attr_name in LABEL_TRIAL_ATTRS:
                if attr_name not in attrs:
                    issues.append(
                        TrialAttrIssue(session_id, partition, path, trial_key, attr_name)
                    )

        if "block_num" in attrs:
            block_nums.add(int(attrs["block_num"]))
        if "n_time_steps" in attrs:
            time_bins.append(int(attrs["n_time_steps"]))
        if "seq_len" in attrs:
            seq_lens.append(int(attrs["seq_len"]))

    has_labels = (
        expects_labels
        and n_trials > 0
        and len(seq_lens) == n_trials
        and all(s > 0 for s in seq_lens)
    )
    total_time_bins = sum(time_bins)

    row = {
        "session_id": session_id,
        "session_date": session_date.isoformat(),
        "partition": partition,
        "n_trials": n_trials,
        "n_blocks": len(block_nums),
        "block_nums": ";".join(str(b) for b in sorted(block_nums)),
        "total_time_bins": total_time_bins,
        "total_minutes": total_time_bins * 0.02 / 60,
        "median_time_bins": statistics.median(time_bins) if time_bins else None,
        "min_time_bins": min(time_bins) if time_bins else None,
        "max_time_bins": max(time_bins) if time_bins else None,
        "median_seq_len": statistics.median(seq_lens) if seq_lens else None,
        "min_seq_len": min(seq_lens) if seq_lens else None,
        "max_seq_len": max(seq_lens) if seq_lens else None,
        "has_labels": has_labels,
    }
    return row, issues


def build_session_manifest(data_root: Path) -> SessionManifestResult:
    """Build the full session manifest by scanning trial attributes only.

    Args:
        data_root: Directory containing per-session subdirectories.

    Returns:
        A `SessionManifestResult` with one manifest row per
        (session, partition) pair, any attribute issues found, the raw
        session listing, and any sessions whose partition-file pattern
        doesn't match the two expected patterns (train-only, or all
        three).
    """
    sessions = discover_sessions(data_root)
    rows: list[dict[str, Any]] = []
    issues: list[TrialAttrIssue] = []
    unexpected_patterns: list[tuple[str, frozenset[str]]] = []

    for session in sessions:
        present = frozenset(session.partitions.keys())
        if present not in EXPECTED_PARTITION_PATTERNS:
            unexpected_patterns.append((session.session_id, present))
        for partition in PARTITIONS:
            if partition not in session.partitions:
                continue
            row, partition_issues = summarize_partition(
                session.session_id,
                session.session_date,
                partition,
                session.partitions[partition],
            )
            rows.append(row)
            issues.extend(partition_issues)

    return SessionManifestResult(
        rows=rows,
        issues=issues,
        sessions=sessions,
        unexpected_partition_patterns=unexpected_patterns,
    )


def write_session_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    """Write manifest rows to a CSV file, creating parent directories.

    Args:
        rows: Manifest rows, as produced by `build_session_manifest`.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def session_date_gaps(sessions: list[SessionFiles]) -> list[tuple[str, date, int | None]]:
    """List sessions in date order with the day-gap since the previous session.

    Args:
        sessions: Sessions to order and diff, as returned by
            `discover_sessions` (re-sorted by date here to be safe).

    Returns:
        `(session_id, session_date, gap_days)` for every session, in date
        order. `gap_days` is the number of days since the previous
        session's date, or `None` for the first session.
    """
    ordered = sorted(sessions, key=lambda s: s.session_date)
    result: list[tuple[str, date, int | None]] = []
    prev_date: date | None = None
    for session in ordered:
        gap_days = None if prev_date is None else (session.session_date - prev_date).days
        result.append((session.session_id, session.session_date, gap_days))
        prev_date = session.session_date
    return result


def iter_trial_ids(
    data_root: Path,
    session_id: str,
    partitions: tuple[str, ...] = ("train", "val"),
) -> Iterator[TrialId]:
    """Yield a `TrialId` for every trial in the given session/partitions.

    Reads only HDF5 attributes (via `iter_trial_attributes`) -- never opens
    `input_features`, `seq_class_ids`, or `transcription`.

    Args:
        data_root: Directory containing per-session subdirectories.
        session_id: Session to enumerate, e.g. `"t15.2023.08.13"`.
        partitions: Which partition files to read, in the given order.
            Defaults to `("train", "val")`, deliberately EXCLUDING `test`
            -- the test partition is unlabeled (`docs/DATASET.md` section
            8), so a caller that wants trials for supervised training or
            PER evaluation should never see test trials by accident. Pass
            `partitions=("test",)` (or include it alongside the others)
            explicitly if test trials are actually wanted, e.g. for
            inference-only inspection. A requested partition whose file
            doesn't exist for this session (e.g. `val` for one of the 4
            train-only sessions) is silently skipped, matching
            `build_session_manifest`'s handling of the same case.

    Yields:
        One `TrialId` per trial group, in HDF5 file iteration order,
        for each requested partition in turn.
    """
    session_date = parse_session_date(session_id)
    session_dir = data_root / session_id
    for partition in partitions:
        path = session_dir / PARTITION_FILENAMES[partition]
        if not path.exists():
            continue
        for trial_key, attrs in iter_trial_attributes(path):
            # seq_len / sentence_label are absent on test-partition trials
            # (unlabeled) -- default to the TrialId placeholder convention
            # (0 / "") documented on TrialId.has_labels, rather than
            # raising or inventing a value.
            yield TrialId(
                session_id=session_id,
                session_date=session_date,
                partition=partition,
                trial_key=trial_key,
                block_num=int(attrs["block_num"]),
                trial_num=int(attrs["trial_num"]),
                n_time_steps=int(attrs["n_time_steps"]),
                seq_len=int(attrs.get("seq_len", 0)),
                sentence_label=str(attrs.get("sentence_label", "")),
            )


_MAX_OPEN_HANDLES = 8
# (pid, file_path) -> open handle. OrderedDict gives us LRU eviction via
# move_to_end() on access and popitem(last=False) to evict the oldest.
# Process-global by design -- see the module docstring's "FILE HANDLE
# MANAGEMENT AND MULTIPROCESSING" section for why the pid is part of the
# key and why handles must never be opened outside a real read call.
_file_cache: "OrderedDict[tuple[int, Path], h5py.File]" = OrderedDict()


def _get_file_handle(path: Path) -> h5py.File:
    """Return an open, read-only handle for `path`, opening it lazily.

    Not part of the public API -- called only by `read_trial`. See the
    module docstring for why this cache is keyed by `(os.getpid(), path)`
    and why handles are never opened eagerly.

    Args:
        path: Partition HDF5 file to open.

    Returns:
        An open `h5py.File`, cached for reuse by later calls in the same
        process. Evicts and closes the least-recently-used handle first if
        this call would exceed `_MAX_OPEN_HANDLES`.
    """
    key = (os.getpid(), path)
    handle = _file_cache.get(key)
    if handle is not None:
        _file_cache.move_to_end(key)  # mark as most-recently-used
        return handle

    handle = h5py.File(path, "r")
    _file_cache[key] = handle
    if len(_file_cache) > _MAX_OPEN_HANDLES:
        _, oldest_handle = _file_cache.popitem(last=False)
        oldest_handle.close()
    return handle


def close_all() -> None:
    """Close every cached file handle opened by the CURRENT process.

    Only closes handles keyed to `os.getpid()`, leaving any (unexpected)
    entries belonging to other processes untouched. Safe to call at any
    time, including when no handles are open. Intended for explicit
    cleanup at the end of a script, or between test cases that read many
    different files.
    """
    current_pid = os.getpid()
    for key in [k for k in _file_cache if k[0] == current_pid]:
        _file_cache.pop(key).close()


def validate_trial_arrays(
    trial_id: TrialId,
    features: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Check that one trial's just-read arrays are internally consistent.

    Args:
        trial_id: The trial `features`/`targets` were read for.
        features: `(n_time_steps, 512)` array, as returned by `read_trial`.
        targets: `(seq_len,)` array, as returned by `read_trial`.

    Raises:
        ValueError: On the first failing check, with `trial_id` included
            in the message. A bare assertion with no context is useless
            when it fires mid-epoch, deep inside a `DataLoader` worker.
            Checks, in order: `features.shape[0] == n_time_steps`;
            `features.shape[1] == 512`; no NaN in `features`; no inf in
            `features`; `targets.shape[0] == seq_len`; every value in
            `targets` is in `[1, 40]` (catches a phoneme-mapping
            off-by-one -- 0 is the CTC blank/pad value and must never
            appear in a real target); `n_time_steps >= seq_len` (CTC's
            hard requirement that input length not be shorter than target
            length).
    """

    def fail(reason: str) -> None:
        raise ValueError(f"validate_trial_arrays: {trial_id} failed: {reason}")

    if features.shape[0] != trial_id.n_time_steps:
        fail(f"features.shape[0]={features.shape[0]} != n_time_steps={trial_id.n_time_steps}")
    if features.shape[1] != 512:
        fail(f"features.shape[1]={features.shape[1]} != 512")
    if np.isnan(features).any():
        fail("features contain NaN")
    if np.isinf(features).any():
        fail("features contain inf")
    if targets.shape[0] != trial_id.seq_len:
        fail(f"targets.shape[0]={targets.shape[0]} != seq_len={trial_id.seq_len}")
    if targets.size and (int(targets.min()) < 1 or int(targets.max()) > 40):
        fail(f"targets out of range [1, 40]: min={int(targets.min())}, max={int(targets.max())}")
    if trial_id.n_time_steps < trial_id.seq_len:
        fail(
            f"n_time_steps={trial_id.n_time_steps} < seq_len={trial_id.seq_len} "
            f"(CTC requires input length >= target length)"
        )


def read_trial(
    data_root: Path,
    trial_id: TrialId,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one trial's neural features and phoneme targets.

    Both arrays are sliced to their true (unpadded) length inside this
    function. `input_features` is stored unpadded already, so slicing to
    `n_time_steps` is a no-op over the whole array; `seq_class_ids` IS
    stored padded to length 500 with the value `0` -- which is also the
    CTC blank index -- so it must be sliced to `seq_len` here, or padding
    would silently masquerade as blank-token targets. Doing both slices
    inside this function, rather than leaving it to each caller, means no
    caller can forget the `seq_class_ids` slice.

    `transcription` is never read -- see the module docstring.

    Args:
        data_root: Directory containing per-session subdirectories.
        trial_id: Which trial to read, as returned by `iter_trial_ids`.
        validate: If True (the default), run `validate_trial_arrays` on
            the result before returning.

    Returns:
        `(features, targets)`:
          - `features`: shape `(n_time_steps, 512)`, dtype `float32` --
            the trial's full neural feature sequence.
          - `targets`: shape `(seq_len,)`, dtype `int32` -- the trial's
            phoneme label sequence. Values are in `[1, 40]`; the CTC
            blank/pad value `0` never appears (see
            `b2t_drift.phonemes`).

    Raises:
        ValueError: If `trial_id.has_labels` is False -- a test-partition
            trial has no real targets, and returning an empty array here
            instead of raising would let a caller silently train or
            evaluate on zero target elements while still seeing a
            computable (misleadingly plausible) loss. Also raised if
            `validate=True` and any check in `validate_trial_arrays`
            fails.
    """
    if not trial_id.has_labels:
        raise ValueError(
            f"read_trial: {trial_id.session_id}/{trial_id.partition}/"
            f"{trial_id.trial_key} has no labels (seq_len=0) -- cannot "
            f"return targets for an unlabeled trial."
        )

    path = data_root / trial_id.session_id / PARTITION_FILENAMES[trial_id.partition]
    group = _get_file_handle(path)[trial_id.trial_key]

    features = group["input_features"][: trial_id.n_time_steps]
    targets = group["seq_class_ids"][: trial_id.seq_len]

    if validate:
        validate_trial_arrays(trial_id, features, targets)

    return features, targets


def read_trials(
    data_root: Path,
    trial_ids: Sequence[TrialId],
    validate: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read multiple trials, reusing one open file handle per source file.

    Trials are grouped by `(session_id, partition)` before reading, so that
    each underlying HDF5 file is read from contiguously rather than
    interleaved with reads from other files -- purely an I/O-locality
    optimization against the LRU handle cache (see the module docstring):
    reading trial_ids in an order that jumps between many source files
    would otherwise risk repeatedly evicting and reopening handles that
    are about to be needed again. Grouping does not affect the output.

    Args:
        data_root: Directory containing per-session subdirectories.
        trial_ids: Trials to read, as returned by `iter_trial_ids`. May
            span multiple sessions and partitions.
        validate: If True, run `validate_trial_arrays` on every trial.
            Defaults to False here (`read_trial`'s own default is True)
            because this is the batch/hot-path entry point a `DataLoader`
            would call every step -- paying per-trial validation cost on
            every step usually isn't worth it once the data is known-good
            from a one-time `validate=True` pass (e.g. the test suite).

    Returns:
        A list of `(features, targets)` tuples, one per input `TrialId`,
        in THE SAME ORDER as `trial_ids` (not file-grouped order -- the
        grouping described above is an internal reading strategy only).
        See `read_trial` for each pair's shape, dtype, and slicing.

    Raises:
        ValueError: If any `trial_id.has_labels` is False, or (when
            `validate=True`) if any trial fails validation. See
            `read_trial`.
    """
    grouped: dict[tuple[str, str], list[tuple[int, TrialId]]] = {}
    for i, trial_id in enumerate(trial_ids):
        grouped.setdefault((trial_id.session_id, trial_id.partition), []).append((i, trial_id))

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for indexed in grouped.values():
        for i, trial_id in indexed:
            results[i] = read_trial(data_root, trial_id, validate=validate)

    return [results[i] for i in range(len(trial_ids))]
