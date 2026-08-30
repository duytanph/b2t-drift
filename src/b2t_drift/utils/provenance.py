"""Run provenance metadata.

Responsible for capturing the information needed to reproduce a training
run: git commit hash, dirty-working-tree flag, resolved config, and
environment details. Consumed by `training/checkpoint.py` when saving
checkpoints.
"""

from __future__ import annotations

from typing import Any


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash, or None if not in a git repo.

    Returns:
        The full git commit SHA, or None.
    """
    raise NotImplementedError


def is_git_dirty() -> bool:
    """Return whether the working tree has uncommitted changes.

    Returns:
        True if there are uncommitted changes, False otherwise.
    """
    raise NotImplementedError


def collect_run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Collect all provenance metadata for a training run.

    Args:
        config: The fully resolved experiment config for this run.

    Returns:
        A dict combining the config with git/environment provenance,
        suitable for storing alongside a checkpoint.
    """
    raise NotImplementedError
