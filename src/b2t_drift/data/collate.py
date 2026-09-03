"""Batch collation and length-bucketed batching for T15 trials.

SCOPE. This module turns a list of `TrialDataset.__getitem__` dicts into a
padded batch, and computes the length metadata `nn.CTCLoss` requires. It
does NOT read files (`hdf5_io.py`), normalize (already done upstream --
`docs/DATASET.md` section 9), or select/filter trials (`splits.py`). A
`collate_fn` or `Sampler` that starts making decisions about which trials
belong in training is exactly the kind of scope creep this project's data
modules are built to avoid.

THIS IS THE HIGHEST-RISK FILE IN THE PIPELINE. Swapping `input_lengths` and
`target_lengths` -- or padding with the wrong value, or getting the target
concatenation order wrong -- produces a model that trains, converges, and
emits garbage without `nn.CTCLoss` ever raising an error: CTC only needs
its length arguments to be self-consistent with the tensors it's given, not
correct. Every assumption `collate_trials` depends on is therefore checked
with an explicit, always-on assertion (see its docstring) rather than
trusted silently.

THE CTC LENGTH CONTRACT. `torch.nn.CTCLoss` takes `log_probs` shaped
`(max_time, batch, n_classes)`, `targets`, `input_lengths` (batch,), and
`target_lengths` (batch,). `targets` may be either a padded 2-D tensor
`(batch, max_target_len)` or, as used here, a 1-D CONCATENATION of every
item's target sequence back to back with no padding at all --
`target_lengths` is what tells CTC where one item's targets end and the
next one's begin. The concatenated form is used here because it is
unambiguous: a padded 2-D target tensor needs a pad value that isn't a real
class, and picking one is one more thing to get right and keep consistent
with the label vocabulary (`b2t_drift.phonemes`); concatenation has no pad
value to choose or misinterpret in the first place.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler

from b2t_drift.data.hdf5_io import TrialId

N_FEATURES = 512
MIN_TARGET_VALUE = 1
MAX_TARGET_VALUE = 40


def collate_trials(items: Sequence[dict]) -> dict:
    """Pad and concatenate a list of `TrialDataset` items into one batch.

    Args:
        items: A batch of dicts, each as returned by
            `TrialDataset.__getitem__` -- must have keys `"features"`
            (`FloatTensor`, `(n_time_steps, 512)`), `"targets"`
            (`LongTensor`, `(seq_len,)`, values in `[1, 40]`),
            `"n_time_steps"` (`int`), `"seq_len"` (`int`),
            `"session_index"` (`int`), `"session_id"` (`str`),
            `"block_num"` (`int`), `"trial_key"` (`str`).

    Returns:
        A dict with keys:
          - `"features"`: `FloatTensor`, shape `(batch, max_time, 512)`.
            Zero-padded in the time dimension up to `max_time` (the
            longest `n_time_steps` in this batch). The pad value 0.0 is
            approximately the per-channel mean, since `input_features`
            is already z-scored (`docs/DATASET.md` section 9) -- NOT
            chosen because it means "no activity". It is not load-bearing:
            `input_lengths` (below) is what actually prevents the model
            from attributing meaning to the padded region; a model that
            attends to time steps past `input_lengths[i]` is already
            wrong regardless of what padding value sits there. Stated
            explicitly so a future reader does not assume otherwise.
          - `"targets"`: `LongTensor`, shape `(sum(target_lengths),)`.
            CONCATENATED, not padded -- see the module docstring for why.
          - `"input_lengths"`: `LongTensor`, shape `(batch,)`. Each item's
            true `n_time_steps`, BEFORE any model-side temporal
            downsampling (a model that downsamples the time axis must
            adjust these itself before passing them to `CTCLoss`; this
            function has no opinion about the model's architecture).
          - `"target_lengths"`: `LongTensor`, shape `(batch,)`. Each
            item's true `seq_len`.
          - `"session_indices"`: `LongTensor`, shape `(batch,)`.
          - `"session_ids"`: `list[str]`, length `batch`.
          - `"block_nums"`: `list[int]`, length `batch`.
          - `"trial_keys"`: `list[str]`, length `batch`.
          `session_ids` / `trial_keys` stay plain Python lists (not
          tensors) for per-session evaluation and error inspection --
          this is exactly why a custom collate function is needed at all;
          the default one cannot batch strings.

    Raises:
        ValueError: If any item has `n_time_steps < seq_len` (CTC's hard
            requirement that input length not be shorter than target
            length), naming the offending `trial_key`; or if any item's
            `targets` contains a value outside `[1, 40]` (0 is the CTC
            blank/pad index and must never appear in a real target),
            naming the offending `trial_key`. These are cheap checks and
            are always on, not gated behind a `validate` flag -- see the
            module docstring for why a length-argument mistake here is
            uniquely dangerous to leave unchecked.
    """
    batch_size = len(items)
    max_time = max(item["n_time_steps"] for item in items)

    for item in items:
        if item["n_time_steps"] < item["seq_len"]:
            raise ValueError(
                f"collate_trials: trial_key={item['trial_key']!r} has "
                f"n_time_steps={item['n_time_steps']} < seq_len={item['seq_len']} "
                f"(CTC requires input length >= target length)."
            )
        targets_i = item["targets"]
        if targets_i.numel():
            min_val = int(targets_i.min())
            max_val = int(targets_i.max())
            if min_val < MIN_TARGET_VALUE or max_val > MAX_TARGET_VALUE:
                raise ValueError(
                    f"collate_trials: trial_key={item['trial_key']!r} has target "
                    f"values outside [{MIN_TARGET_VALUE}, {MAX_TARGET_VALUE}]: "
                    f"min={min_val}, max={max_val}."
                )

    features = torch.zeros((batch_size, max_time, N_FEATURES), dtype=torch.float32)
    input_lengths = torch.empty(batch_size, dtype=torch.int64)
    target_lengths = torch.empty(batch_size, dtype=torch.int64)
    session_indices = torch.empty(batch_size, dtype=torch.int64)
    session_ids: list[str] = []
    block_nums: list[int] = []
    trial_keys: list[str] = []
    target_chunks: list[torch.Tensor] = []

    for i, item in enumerate(items):
        t = item["n_time_steps"]
        features[i, :t, :] = item["features"]
        input_lengths[i] = t
        target_lengths[i] = item["seq_len"]
        session_indices[i] = item["session_index"]
        session_ids.append(item["session_id"])
        block_nums.append(item["block_num"])
        trial_keys.append(item["trial_key"])
        target_chunks.append(item["targets"])

    targets = torch.cat(target_chunks) if target_chunks else torch.empty(0, dtype=torch.int64)

    # Always-on sanity checks -- see the module and class docstrings for
    # why a silent length mistake here is worse than an assertion cost.
    assert targets.numel() == int(target_lengths.sum()), (
        "collate_trials: concatenated targets length does not match "
        "target_lengths.sum() -- this should be unreachable."
    )
    assert features.shape[1] == int(input_lengths.max()), (
        "collate_trials: features time dimension does not match "
        "input_lengths.max() -- this should be unreachable."
    )

    return {
        "features": features,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "session_indices": session_indices,
        "session_ids": session_ids,
        "block_nums": block_nums,
        "trial_keys": trial_keys,
    }


class LengthBucketedSampler(Sampler[list[int]]):
    """Batches trials of similar duration together, to cut padding waste.

    Trial durations span roughly 7.4x (138 to 2179+ bins,
    `docs/DATASET.md` section 6). A randomly-composed batch pads every
    short trial out to the longest trial that happened to land in that
    batch, wasting most of the compute on `nn.GRU` inputs and `CTCLoss`
    time steps that are pure padding. Sorting trials by length before
    chunking into batches groups similar durations together instead.

    TRADEOFF, recorded here and in `docs/DESIGN.md` as a deliberate
    choice: bucketing reduces padding waste but makes batches non-i.i.d.
    in sentence length -- a given batch is drawn from a narrow length
    band, not a uniform random sample of the dataset's length
    distribution. This is standard practice in speech recognition
    training and is considered an acceptable tradeoff, not a bug; it is
    called out explicitly because a batch sampler that quietly changes
    what "one batch" statistically represents is exactly the kind of
    thing that should be documented, not just discovered by whoever
    reads the training curves later.

    Method: sort trial indices by `n_time_steps` (with a small amount of
    per-epoch random noise added to the sort key -- see `bucket_noise`
    below), chunk the sorted order into batches of `batch_size`, then
    shuffle the ORDER of batches (never their contents). Yields lists of
    dataset indices -- use as a `DataLoader`'s `batch_sampler` argument,
    never as its `sampler` argument (a plain `sampler` yields one index
    at a time; a `batch_sampler` yields a whole batch's worth at once,
    which is what this class does).

    `__iter__` IS PURE: it reads `self._epoch` but never modifies it.
    **A training loop MUST call `set_epoch(epoch)` itself, once per
    epoch.** Without it, every epoch gets the exact same ordering and
    bucket composition -- `__iter__` used to advance an internal counter
    automatically on every call so this "just worked" without
    `set_epoch`, but that made two harmless-looking operations (calling
    `iter()` twice on the same sampler, or inspecting it) silently change
    its future behavior. State changes now happen only where they're
    visible: in `set_epoch`.
    """

    def __init__(
        self,
        trial_ids: Sequence[TrialId],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        bucket_noise: float = 0.1,
        drop_last: bool = False,
    ) -> None:
        """Build a length-bucketed batch sampler over `trial_ids`.

        Args:
            trial_ids: The trials to batch, in the SAME order as the
                `Dataset` this sampler will be paired with (indices
                yielded by this sampler are positions into `trial_ids`,
                which must line up with the `Dataset`'s own indexing).
            batch_size: Trials per batch.
            shuffle: If True (the default), shuffle the ORDER of batches
                each epoch. Never shuffles which trials share a batch --
                that composition is fixed by the length sort (plus
                `bucket_noise`).
            seed: Base RNG seed. Combined with the epoch set via
                `set_epoch` so that batch order and bucket-noise
                composition are both reproducible for a given
                `(seed, epoch)` pair.
            bucket_noise: Fraction of a trial's own `n_time_steps` used
                as the scale of random jitter added to its sort key
                (uniform in `[-bucket_noise, +bucket_noise] *
                n_time_steps`). Without this, the same trials would be
                batched together every single epoch (the length sort is
                deterministic) whenever the epoch DOES change, which
                correlates gradient noise across epochs -- two epochs
                would see literally the same per-batch composition, just
                possibly in a different order. `0.1` means each trial's
                sort position can drift by up to +/-10% of its own
                length, enough to shuffle batch membership near bucket
                boundaries without destroying the overall
                length-similarity grouping.
            drop_last: If True, drop a final batch that has fewer than
                `batch_size` trials rather than yielding it short.
                **Recommended True for training**: at `batch_size=32`
                against the 3,969-trial TRAIN block (`docs/DATASET.md`
                section 13), `3969 = 124*32 + 1` -- the last batch of an
                epoch would contain exactly 1 trial, which gives an
                unusually noisy gradient and would break any layer that
                relies on batch statistics (e.g. batch normalization,
                not currently used here, but a future addition could add
                one without knowing this constraint exists). **Required
                False (the default) for evaluation**: every held-out
                trial must be scored, including whichever ones would
                fall in a short final batch -- dropping trials from PER
                evaluation silently biases the metric.
        """
        self.trial_ids = tuple(trial_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_noise = bucket_noise
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the NEXT `__iter__` call's ordering.

        Args:
            epoch: Epoch number. Combined with `seed` to derive the RNG
                state `__iter__` will use, so `set_epoch(0)` followed by
                `set_epoch(1)` reproducibly gives two DIFFERENT batch
                orderings/compositions, and calling `set_epoch(0)` again
                later reproduces the first one exactly.

        Follows the same convention as
        `torch.utils.data.distributed.DistributedSampler.set_epoch`.
        **This is the ONLY way this sampler's ordering changes across
        epochs** -- `__iter__` itself is pure (see the class docstring).
        A training loop that never calls this will silently reuse epoch
        0's ordering and bucket composition forever; call it once at the
        start of every epoch, before iterating the `DataLoader`.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch's worth of batches, each a list of indices.

        Pure with respect to `self._epoch`: reads it, does not modify
        it. Calling this twice in a row without an intervening
        `set_epoch` yields the SAME batches both times (see the class
        docstring for why this matters).

        Yields:
            Lists of indices into `trial_ids`, one list per batch, in
            batch-size chunks of at most `self.batch_size`. The final,
            possibly-short chunk is omitted if `self.drop_last` is True.
        """
        rng = random.Random(self.seed + self._epoch)

        n = len(self.trial_ids)
        if self.bucket_noise:
            keys = [
                trial_id.n_time_steps
                + rng.uniform(-self.bucket_noise, self.bucket_noise) * trial_id.n_time_steps
                for trial_id in self.trial_ids
            ]
        else:
            keys = [trial_id.n_time_steps for trial_id in self.trial_ids]

        order = sorted(range(n), key=lambda i: keys[i])
        batches = [order[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        """Number of batches this sampler yields per epoch."""
        n = len(self.trial_ids)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
