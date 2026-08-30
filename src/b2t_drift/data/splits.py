"""Temporal session splits.

Responsible for assigning sessions to train/val/test splits based on their
recording date, and for reading/writing versioned split manifests under
`splits/`. Splits must never shuffle sessions across time -- this is a
temporal-generalization study, not an i.i.d. one.
"""

from __future__ import annotations

from pathlib import Path


def build_temporal_splits(
    session_ids: list[str],
    session_dates: list[str],
) -> dict[str, list[str]]:
    """Assign sessions to train/val/test splits by recording date.

    Args:
        session_ids: Session identifiers.
        session_dates: ISO-format recording dates, one per session id.

    Returns:
        A dict mapping split name ("train", "val", "test") to a list of
        session ids, ordered by date within each split.
    """
    raise NotImplementedError


def write_split_manifest(splits: dict[str, list[str]], path: Path) -> None:
    """Write a split assignment to a versioned JSON manifest file.

    Args:
        splits: Split name -> session ids, as returned by
            `build_temporal_splits`.
        path: Destination path under `splits/`.
    """
    raise NotImplementedError


def read_split_manifest(path: Path) -> dict[str, list[str]]:
    """Read a previously written split manifest.

    Args:
        path: Path to a JSON manifest under `splits/`.

    Returns:
        Split name -> session ids.
    """
    raise NotImplementedError
