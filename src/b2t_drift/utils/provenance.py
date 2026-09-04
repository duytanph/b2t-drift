"""Run provenance metadata.

Responsible for capturing the information needed to judge whether a
training run is reproducible: git commit hash, dirty-working-tree flag,
and environment details (torch/CUDA/GPU/Python/platform, installed package
versions). Written to `run_dir/provenance.json` by `scripts/02_train.py`,
alongside (not inside) the checkpoint `training/checkpoint.py` saves --
provenance describes the ENVIRONMENT a run happened in, while a
checkpoint's own `config` field describes the run's PARAMETERS; the two
answer different questions and are kept as separate artifacts on purpose.

WHY A DIRTY WORKING TREE MUST BE RECORDED. The git commit hash alone is
not enough: `git rev-parse HEAD` reports the last COMMITTED state, but a
run's actual behavior comes from whatever is on disk at the moment it
runs, including any uncommitted edits. A run launched from a dirty tree is
not reproducible from the commit hash alone -- re-checking out that commit
later reproduces a DIFFERENT (the last-committed) version of the code, not
what actually ran. `capture_provenance` therefore always records
`git_dirty` alongside `git_commit`, so a reader of `provenance.json` can
see this limitation stated plainly rather than discover it the hard way
when a "reproduction" run doesn't match.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from typing import Any

import torch

# Optional experiment trackers whose installed version (if present) is
# worth recording -- see `training/logging.py`'s module docstring for why
# there are three of these and why all are optional.
_TRACKER_PACKAGES = ("wandb", "tensorboard", "mlflow")

# Packages this project always depends on (`pyproject.toml`'s base
# `dependencies`) whose versions matter for reproducing a run's numerics.
_CORE_PACKAGES = ("torch", "numpy", "h5py")


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash, or None if not in a git repo.

    Returns:
        The full git commit SHA, or `None` if `git` isn't installed, this
        isn't a git repository, or the call fails for any other reason --
        matching `data/splits.py`'s `_git_commit_hash` convention (a
        provenance capture must not itself crash a run).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def is_git_dirty() -> bool | None:
    """Return whether the working tree has uncommitted changes.

    Returns:
        `True` if `git status --porcelain` reports any changes (staged,
        unstaged, or untracked files), `False` if it reports none, or
        `None` if this could not be determined at all (`git` unavailable
        or not a git repository) -- deliberately NOT defaulted to `True`
        or `False` in that case, since either default would silently
        assert something we don't actually know; `None` is recorded as
        `None` in `provenance.json` rather than papered over.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _installed_version(package: str) -> str | None:
    """Installed version of `package`, or None if not installed."""
    if importlib_util.find_spec(package) is None:
        return None
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        # find_spec succeeded (module importable) but no distribution
        # metadata exists -- e.g. a vendored or editable-without-metadata
        # install. Record that it's present but the version is unknown,
        # rather than claiming it's absent.
        return "unknown"


def capture_provenance(seed: int | None = None) -> dict[str, Any]:
    """Capture everything needed to judge this run's reproducibility.

    Args:
        seed: The RNG seed this run used (`utils.seeding.set_seed`'s
            argument), if any -- recorded here since it's as much a part
            of "what would it take to reproduce this run" as the code
            version is, and there is no other natural home for it in a
            provenance record.

    Returns:
        A dict with keys: `git_commit`, `git_dirty`, `torch_version`,
        `cuda_version` (`None` if CUDA is unavailable), `gpu_name`
        (`None` if CUDA is unavailable), `python_version`, `platform`,
        `timestamp` (ISO 8601, UTC), `seed`, and `package_versions` (a
        nested dict: `torch`, `numpy`, `h5py`, plus `wandb`/`tensorboard`/
        `mlflow` for whichever of those three are actually installed --
        an absent tracker is simply not a key in `package_versions`,
        rather than `None`, so the dict itself shows which trackers this
        run could have used).
    """
    cuda_available = torch.cuda.is_available()

    package_versions = {pkg: _installed_version(pkg) for pkg in _CORE_PACKAGES}
    for pkg in _TRACKER_PACKAGES:
        version = _installed_version(pkg)
        if version is not None:
            package_versions[pkg] = version

    return {
        "git_commit": get_git_commit_hash(),
        "git_dirty": is_git_dirty(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": seed,
        "package_versions": package_versions,
    }
