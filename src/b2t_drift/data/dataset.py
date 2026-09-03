"""PyTorch Dataset for T15 neural/phoneme trial pairs.

Wraps a list of `TrialId` (from `b2t_drift.data.hdf5_io`) in a
`torch.utils.data.Dataset`, returning per-trial tensors.

SCOPE. This module does NOT normalize -- the release pipeline already
z-scores `input_features` upstream (`docs/DATASET.md` section 9); does NOT
pad or batch trials of different lengths together (that's
`data/collate.py`'s job, needed precisely because `TrialDataset` returns
variable-length tensors the default collate function cannot stack); and
does NOT filter, select, or split trials (that's `data/splits.py`'s job --
`TrialDataset` takes whatever `TrialId`s it's handed and has no opinion
about which ones those should be). Keeping this dumb on purpose: filtering
logic inside a `Dataset` is exactly where silent train/eval mismatches come
from -- a `Dataset` that quietly drops or reorders trials by some internal
rule makes it possible for two datasets built from "the same" `trial_ids`
to actually disagree about what they contain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from b2t_drift.data.hdf5_io import TrialId, read_trial


class TrialDataset(Dataset):
    """One torch Dataset item per trial, backed by `hdf5_io.read_trial`.

    Arrays are read lazily in `__getitem__`, never in `__init__` -- see
    `__getitem__`'s docstring for why that matters under
    `DataLoader(num_workers > 0)`.
    """

    def __init__(
        self,
        data_root: Path,
        trial_ids: Sequence[TrialId],
        session_to_index: Mapping[str, int],
        validate: bool = False,
    ) -> None:
        """Build the dataset from an explicit list of trials.

        Args:
            data_root: Directory containing per-session subdirectories,
                passed straight through to `hdf5_io.read_trial`.
            trial_ids: The trials this dataset will serve, in order.
                Stored as a `tuple` -- for the full 9,498-labeled-trial
                dataset this is a negligible amount of memory (small
                dataclasses, no arrays), so there is no reason for
                anything more elaborate than eager storage here. Which
                trials these are, and in what order, is entirely
                `splits.py`'s decision; this class never filters,
                reorders, or deduplicates them.
            session_to_index: Maps each trial's `session_id` to a
                contiguous integer, needed by the Week 3 day-specific
                input layers (one per session). Passed in rather than
                derived from `trial_ids` so the SAME mapping can be
                shared across a train dataset and an eval dataset built
                from a disjoint set of sessions -- deriving it locally
                from each dataset's own `trial_ids` would give two
                datasets different, incompatible index spaces for the
                same session.
            validate: Passed straight through to `hdf5_io.read_trial` on
                every `__getitem__` call. Defaults to False, matching
                `read_trial`'s own hot-path default; set True for a
                one-time correctness pass, not routine training.

        Raises:
            ValueError: If any `trial_id.session_id` in `trial_ids` is
                missing from `session_to_index` -- a stale or incomplete
                mapping is a configuration error that should fail
                immediately at construction, not surface as a `KeyError`
                deep inside a later training step. Also raised if any
                trial in `trial_ids` has `has_labels` False: an unlabeled
                (test-partition) trial has no targets `__getitem__` could
                return, and this dataset has no way to represent "no
                target" other than refusing to hold the trial at all --
                filter such trials out in `splits.py` before constructing
                this dataset.
        """
        missing_sessions = {t.session_id for t in trial_ids} - set(session_to_index)
        if missing_sessions:
            raise ValueError(
                f"TrialDataset: session_to_index is missing session_id(s) "
                f"{sorted(missing_sessions)} that are present in trial_ids."
            )
        unlabeled = [t for t in trial_ids if not t.has_labels]
        if unlabeled:
            raise ValueError(
                f"TrialDataset: {len(unlabeled)} trial(s) have has_labels=False "
                f"(e.g. {unlabeled[0]}) -- this dataset only serves labeled "
                f"trials; filter test-partition trials out in splits.py "
                f"before constructing it."
            )

        self.data_root = data_root
        self.trial_ids = tuple(trial_ids)
        self.session_to_index = session_to_index
        self.validate = validate

    def __len__(self) -> int:
        """Number of trials this dataset serves."""
        return len(self.trial_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        """Read and return one trial.

        File handles are opened lazily, the first time a trial from a
        given HDF5 file is actually read, via `hdf5_io`'s pid-keyed LRU
        handle cache -- never in `__init__` or anywhere else that could
        run before `DataLoader` worker processes are forked/spawned. This
        is what makes this dataset safe to use with
        `DataLoader(num_workers > 0)`: each worker process only ever
        opens (and caches) handles under its own pid, so a handle opened
        in the parent process is never shared across a process boundary.
        See `b2t_drift.data.hdf5_io`'s module docstring for the full
        explanation of why sharing a handle across processes would
        otherwise corrupt reads or hang.

        Args:
            idx: Index into this dataset, in `[0, len(self))`.

        Returns:
            A dict with keys:
              - `"features"`: `FloatTensor`, shape `(n_time_steps, 512)`.
              - `"targets"`: `LongTensor` (`int64`), shape `(seq_len,)`,
                values in `[1, 40]`. `int64` specifically because
                `torch.nn.CTCLoss` requires it.
              - `"n_time_steps"`: `int`.
              - `"seq_len"`: `int`.
              - `"session_index"`: `int`, this trial's session mapped
                through `session_to_index`.
              - `"session_id"`: `str`.
              - `"block_num"`: `int`.
              - `"trial_key"`: `str`.
            `session_id` and `trial_key` are kept as plain strings (not
            tensors) for per-session evaluation and error inspection --
            the default `collate_fn` cannot batch strings, which is
            exactly why `data/collate.py` needs a custom one rather than
            relying on the default.
        """
        trial_id = self.trial_ids[idx]
        features, targets = read_trial(self.data_root, trial_id, validate=self.validate)

        return {
            "features": torch.from_numpy(features).float(),
            "targets": torch.from_numpy(targets).long(),
            "n_time_steps": trial_id.n_time_steps,
            "seq_len": trial_id.seq_len,
            "session_index": self.session_to_index[trial_id.session_id],
            "session_id": trial_id.session_id,
            "block_num": trial_id.block_num,
            "trial_key": trial_id.trial_key,
        }
