"""Unit tests for `b2t_drift.utils.provenance`."""

from __future__ import annotations

import re

from b2t_drift.utils.provenance import (
    capture_provenance,
    get_git_commit_hash,
    is_git_dirty,
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_get_git_commit_hash_returns_a_full_sha() -> None:
    # This project's own working directory IS a git repo.
    commit_hash = get_git_commit_hash()
    assert commit_hash is not None
    assert _SHA_RE.match(commit_hash), commit_hash


def test_is_git_dirty_returns_a_bool_in_this_repo() -> None:
    dirty = is_git_dirty()
    assert dirty is None or isinstance(dirty, bool)


def test_capture_provenance_has_all_required_keys() -> None:
    provenance = capture_provenance(seed=42)
    expected_keys = {
        "git_commit", "git_dirty", "torch_version", "cuda_version", "gpu_name",
        "python_version", "platform", "timestamp", "seed", "package_versions",
    }
    assert set(provenance.keys()) == expected_keys
    assert provenance["seed"] == 42


def test_capture_provenance_records_dirty_working_tree_not_just_commit() -> None:
    """The specific requirement: git_dirty must be present and distinct
    from git_commit, so a reader can tell a dirty run apart from a clean
    one without re-deriving it."""
    provenance = capture_provenance()
    assert "git_dirty" in provenance
    assert "git_commit" in provenance
    assert provenance["git_dirty"] is None or isinstance(provenance["git_dirty"], bool)


def test_capture_provenance_package_versions_include_core_deps() -> None:
    provenance = capture_provenance()
    versions = provenance["package_versions"]
    assert versions.get("torch")
    assert versions.get("numpy")
    assert versions.get("h5py")


def test_capture_provenance_tensorboard_version_present_when_installed() -> None:
    # tensorboard IS installed in this environment (pyproject.toml's
    # `tensorboard` extra) -- confirms an installed tracker's version is
    # actually recorded, not just core packages.
    provenance = capture_provenance()
    assert "tensorboard" in provenance["package_versions"]


def test_capture_provenance_no_cuda_gives_none_gpu_fields() -> None:
    import torch

    provenance = capture_provenance()
    if not torch.cuda.is_available():
        assert provenance["cuda_version"] is None
        assert provenance["gpu_name"] is None
