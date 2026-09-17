"""Model checkpointing.

Responsible for saving and loading model/optimizer state to `runs/`, along
with the metadata needed to reproduce a run (metrics at that point, and the
resolved config). Provenance (git commit, environment) is a SEPARATE
concern, written alongside a checkpoint as `provenance.json` by
`utils.provenance.capture_provenance` -- this module only handles the
model/optimizer state and the two small dicts a checkpoint always carries.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

# Substrings that identify PyTorch's `weights_only=True` unpickler failing
# specifically because of numpy's `numpy.core` -> `numpy._core` module
# rename (numpy >= 2.0), rather than some other, unrelated load failure --
# see `load_checkpoint`'s docstring for why this distinction matters.
_NUMPY_CORE_RENAME_MARKERS = ("numpy.core", "numpy._core")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Save model/optimizer state and run metadata to disk.

    Args:
        path: Destination checkpoint file path, under `runs/`. Parent
            directories are created if missing.
        model: Model whose `state_dict()` should be saved.
        optimizer: Optimizer whose `state_dict()` should be saved.
        epoch: Current epoch number.
        metrics: This checkpoint's metrics (e.g. `{"val_per": 0.31,
            "train_loss": 1.2, ...}`) -- whatever `training/loop.py`'s
            `fit` computed for the epoch this checkpoint represents.
        config: The fully resolved experiment config for this run, saved
            alongside the weights so a checkpoint is self-describing --
            reloading it later does not depend on separately tracking
            down which config file (or which since-edited version of one)
            produced it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_torch_checkpoint_with_numpy_fallback(
    path: Path,
    map_location: str = "cpu",
    caller: str = "load_checkpoint",
) -> dict[str, Any]:
    """`torch.load` a checkpoint file, with an explicit, narrow fallback for
    the numpy `numpy.core`/`numpy._core` rename incompatibility.

    `torch.load` is tried first with `weights_only=True` (the safer mode:
    it refuses to unpickle arbitrary objects, only tensors and a small
    allowlist of types). This has been observed to FAIL on checkpoints
    containing numpy scalars/arrays pickled under one numpy major version
    and loaded under another, because numpy 2.0 renamed its internal
    `numpy.core` module to `numpy._core` and `weights_only=True`'s
    unpickler does not know how to resolve the old (or new) name for the
    other version -- surfacing as a `pickle`/`torch.serialization` error
    mentioning `numpy.core` or `numpy._core`. When (and ONLY when) the
    failure matches that specific signature, this function retries with
    `weights_only=False` -- EXPLICITLY, with a `UserWarning` naming what
    happened and why, never as a silent blanket fallback. Any OTHER
    `torch.load` failure propagates unchanged: silently swallowing an
    unrelated load error would be exactly the kind of hidden failure mode
    this project's validation philosophy exists to avoid (`CLAUDE.md`,
    `data/splits.py`'s module docstring).

    Shared by `load_checkpoint` (this project's own checkpoints) and
    `models.pretrained_baseline.load_pretrained_baseline_checkpoint` (the
    externally-sourced official checkpoint, the file this fallback was
    originally written for) so the hazard and its handling are recorded
    in exactly one place.

    Args:
        path: Checkpoint file path to load.
        map_location: Passed straight through to `torch.load`.
        caller: Name to attribute the warning/exception messages to, so a
            reader sees which caller triggered the fallback.

    Returns:
        Whatever `torch.load` returns (a dict, for every checkpoint format
        this project reads).
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        message = str(exc)
        if any(marker in message for marker in _NUMPY_CORE_RENAME_MARKERS):
            warnings.warn(
                f"{caller}: torch.load(weights_only=True) failed on "
                f"{path} with what looks like the known numpy "
                f"'numpy.core'/'numpy._core' module-rename incompatibility "
                f"({exc}). Retrying with weights_only=False -- this trusts "
                f"the checkpoint's full pickle stream, which is safe for a "
                f"checkpoint this project wrote itself, or a checkpoint "
                f"from a known, documented source, but would NOT be safe "
                f"for an untrusted file. Falling back explicitly, not "
                f"silently.",
                stacklevel=2,
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load model (and optionally optimizer) state from a checkpoint file.

    See `load_torch_checkpoint_with_numpy_fallback` for the `weights_only`
    fallback this relies on.

    Args:
        path: Checkpoint file path to load.
        model: Model to load state into, in place (`load_state_dict`,
            strict by default -- a shape/key mismatch raises immediately
            rather than silently loading a partial or wrong state).
        optimizer: Optional optimizer to load state into, in place.
        map_location: Passed straight through to `torch.load` -- e.g.
            `"cpu"` (the default, always loadable regardless of what
            device saved the checkpoint) or `"cuda"`.

    Returns:
        A dict with keys `"epoch"`, `"metrics"`, and `"config"`, exactly
        as `save_checkpoint` wrote them (model/optimizer state is loaded
        in place, not returned again).

    Raises:
        ValueError: If `optimizer` is given but the checkpoint has no
            `"optimizer_state_dict"` key -- an optimizer-less checkpoint
            (there is currently no code path that produces one, since
            `save_checkpoint` always includes it) loaded with an
            optimizer present is a mismatch worth surfacing explicitly
            rather than silently leaving the optimizer unchanged.
    """
    path = Path(path)
    checkpoint = load_torch_checkpoint_with_numpy_fallback(
        path, map_location=map_location, caller="load_checkpoint",
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(
                f"load_checkpoint: {path} has no 'optimizer_state_dict', "
                f"but an optimizer was given to load state into."
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
        "config": checkpoint["config"],
    }
