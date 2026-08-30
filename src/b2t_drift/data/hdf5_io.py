"""Low-level HDF5 reading for the T15 dataset.

Responsible for discovering session/partition HDF5 files under
`data/raw/hdf5_data_final/` and reading trial-group data out of them.
Nothing above this module should touch `h5py` directly.

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
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Iterator
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

    TODO: not implemented. Not needed by `scripts/00_verify_data.py`, which
    must read only HDF5 attributes -- the full dataset is ~12 GB. Implement
    this when `data/dataset.py` needs to load real trial data.

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
