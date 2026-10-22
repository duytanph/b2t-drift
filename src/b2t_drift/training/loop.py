"""Training and validation loop.

Responsible for the epoch-level train/eval loop: iterating batches (as
produced by `collate_trials`, never `collate_batch` -- that name never
existed in this codebase and was a stale docstring reference), computing
CTC loss, stepping the optimizer, and logging metrics. Called by
`scripts/02_train.py`; contains no CLI or config-parsing logic itself
(`CLAUDE.md`).

CRITICAL: CTC LOSS MUST RUN IN FP32, EVEN UNDER AMP. Automatic mixed
precision (`torch.autocast(dtype=torch.float16)`) speeds up the GRU
forward pass, but `nn.CTCLoss` computes a log-sum-exp over every possible
alignment between input and target -- in fp16, that sum UNDERFLOWS
silently. This is NOT a crash: training proceeds, the loss looks like a
plausible number, and the model quietly gets worse gradients than it
would in fp32, with nothing here raising an error. `train_one_epoch`
therefore runs the model's forward pass inside `torch.autocast` (for
speed) but casts `log_probs` to `float32` IMMEDIATELY AFTER, OUTSIDE the
autocast context, before `criterion(...)` ever sees it -- CTC loss itself
always runs in fp32 regardless of whether AMP is enabled for the forward
pass. Get this backwards (compute the loss inside the autocast block) and
nothing crashes; the degradation is silent, exactly the failure mode this
project's validation philosophy exists to catch (`CLAUDE.md`'s "known
pitfalls").

WHY `ctc_input_lengths`, NOT `input_lengths` DIRECTLY, FOR THE CTC LOSS.
`model.ctc_input_lengths(input_lengths)` is the single place a future
temporal-downsampling model would divide lengths before `nn.CTCLoss`
(`models/gru_ctc.py`'s module docstring). Calling `criterion` with raw
`input_lengths` instead would still work today (this version of `GRUCTC`
has no downsampling, so `ctc_input_lengths` is currently the identity),
but would silently stop being correct the moment downsampling is added,
at exactly the call site someone would have no reason to revisit. Always
route through it, even though it's a no-op right now.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from b2t_drift.data.collate import LengthBucketedSampler
from b2t_drift.evaluation.ctc_decode import greedy_ctc_decode
from b2t_drift.evaluation.edit_distance import (
    EditCounts,
    levenshtein_alignment_with_deleted_tokens,
)
from b2t_drift.evaluation.metrics import aggregate_error_rate
from b2t_drift.phonemes import SILENCE_INDEX
from b2t_drift.training.checkpoint import save_checkpoint
from b2t_drift.training.logging import ConsoleLogger, CSVLogger, MultiLogger, RunLogger


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """A new dict with `collate_trials`'s tensor values moved to `device`.

    Non-tensor values (`"session_ids"`, `"block_nums"`, `"trial_keys"` --
    plain Python lists, `data/collate.py`'s schema) are copied through
    unchanged; they stay on CPU because they're never on a device to
    begin with. Returns a NEW dict rather than mutating `batch` in place,
    so a caller that also needs the original (CPU-resident) tensors for
    something else -- `evaluate` splits `targets` back into per-item
    sequences using the original CPU tensors, for instance -- still has
    them.
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
    epoch: int = 0,
    sampler: LengthBucketedSampler | None = None,
    loggers: RunLogger | None = None,
    day_layer_penalty: float = 0.0,
) -> dict[str, Any]:
    """Run one training epoch.

    Args:
        model: The model being trained (expected: `models.gru_ctc.GRUCTC`,
            for its `ctc_input_lengths` and `(features, session_indices,
            input_lengths)` forward signature -- see the module
            docstring).
        dataloader: Yields batches as produced by `collate_trials`.
        optimizer: Optimizer for `model`'s parameters.
        criterion: Loss function, expected to be `nn.CTCLoss` constructed
            with `zero_infinity=True` (a `UserWarning` is emitted if it
            looks like a `nn.CTCLoss` with `zero_infinity=False`, since
            that turns "INFINITE LOSS" below from a counted, survivable
            event into an uncaught `inf`/`nan` propagating into every
            downstream gradient).
        device: Device to run computation on.
        scaler: A `torch.amp.GradScaler`, or `None` (the default) to
            train in full precision. When given, the forward pass runs
            inside `torch.autocast` and gradients are scaled/unscaled
            around the backward pass and clip -- see the module
            docstring's CTC-in-fp32 warning and "GRADIENT CLIPPING" below
            for why the unscale step specifically cannot be skipped.
        grad_clip: Max gradient norm for `clip_grad_norm_`, applied to
            EVERY step (AMP or not) right before `optimizer.step()`.
            GRUs trained with CTC produce occasional enormous gradients;
            one unclipped step can destroy an otherwise-converged model.
            Default 1.0 is a conventional starting point, not tuned for
            this project specifically.
        epoch: Current epoch number. Used only to call
            `sampler.set_epoch(epoch)` when `sampler` is a
            `LengthBucketedSampler` -- see `sampler`'s docstring entry.
        sampler: If this is a `LengthBucketedSampler`,
            `sampler.set_epoch(epoch)` is called before iterating
            `dataloader`. WITHOUT this, `LengthBucketedSampler` silently
            reuses epoch 0's batch composition and ordering FOREVER (its
            own docstring's explicit warning) -- every epoch would train
            on the exact same batches in the exact same order, which
            defeats the point of training for more than one epoch. `None`
            (the default) is fine for a plain `Sampler`/no sampler, which
            has no such statefulness to advance.
        loggers: Optional `RunLogger`-shaped object (e.g. a
            `training.logging.MultiLogger`). If given, `log_scalars` is
            called once per (non-skipped) batch with a running
            `{"batch_loss": ...}`, giving a fine-grained loss curve
            distinct from `fit`'s once-per-epoch summary logging.
        day_layer_penalty: Weight (lambda) for
            `model.day_layers.identity_penalty()`, added to the CTC loss
            BEFORE `.backward()` on every step -- `loss = ctc_loss +
            day_layer_penalty * model.day_layers.identity_penalty()`.
            Default `0.0`: no penalty, byte-for-byte the same training
            objective as before this existed. A no-op (not even computed)
            when `model.day_layers is None` (e.g. `use_day_layers=False`)
            regardless of this value, since there is nothing to
            regularize. See `DayLayers.identity_penalty`'s docstring for
            why this anchors toward IDENTITY rather than zero (unlike
            `weight_decay`, which is the wrong regularizer for a
            per-session correction whose no-op point is the identity
            transform, not the zero matrix -- `docs/FINDINGS.md`).

    Returns:
        A dict with keys:
          - `"loss"`: trial-count-weighted average loss over batches
            whose loss was finite (excludes the batches counted in
            `"n_inf_batches"` below -- `nan` if EVERY batch in the epoch
            was non-finite, with a `UserWarning`).
          - `"n_batches"`: total batches iterated this epoch.
          - `"n_inf_batches"`: batches with at least one non-finite
            RAW per-item loss (checked independently of `criterion`'s own
            `zero_infinity`, which replaces an infinite per-item loss
            with 0 before the batch's reduced scalar loss is ever
            returned -- so checking THAT scalar for `inf`/`nan` would
            essentially never fire, confirmed directly: an infeasible
            single-item batch returns exactly `0.0`, not `inf`, under
            `zero_infinity=True`). SKIPPED for backward/step (never train
            from a broken gradient) but explicitly counted rather than
            silently zeroed and forgotten -- `zero_infinity=True` is what
            makes it SAFE to keep training past one, but zeroing without
            counting would hide a real, possibly-worsening problem (e.g.
            AMP instability -- see the module docstring).
          - `"grad_norm_mean"`: mean of `clip_grad_norm_`'s return value
            (the PRE-clip total gradient norm, which is the informative
            number -- the post-clip norm is `<= grad_clip` by
            definition) over finite-loss batches whose grad_norm was
            ALSO finite (`nan` if none were -- see the AMP note in the
            source: under AMP, an individual batch's UNSCALED gradient
            can briefly overflow to `inf` even when its loss was fine --
            observed directly in this project's own smoke test during
            GradScaler's first few steps -- and one `inf` in a plain mean
            would silently make the WHOLE epoch's `grad_norm_mean`
            report `inf`, hiding every other batch's real value).
          - `"seconds"`: wall-clock time for this call.
    """
    if isinstance(sampler, LengthBucketedSampler):
        sampler.set_epoch(epoch)

    if isinstance(criterion, torch.nn.CTCLoss) and not criterion.zero_infinity:
        warnings.warn(
            "train_one_epoch: criterion is an nn.CTCLoss with "
            "zero_infinity=False. An infinite per-item loss will not be "
            "zeroed, corrupting this batch's gradient and (once it's "
            "included in the running loss) every batch after it that "
            "shares the same corrupted parameters. Construct criterion "
            "with zero_infinity=True.",
            stacklevel=2,
        )

    model.train()
    start = time.perf_counter()

    total_loss = 0.0
    total_trials = 0
    grad_norms: list[float] = []
    n_batches = 0
    n_inf_batches = 0
    global_step_base = epoch * len(dataloader)

    for batch_idx, raw_batch in enumerate(dataloader):
        n_batches += 1
        batch = _move_batch_to_device(raw_batch, device)
        batch_size = batch["features"].shape[0]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                log_probs = model(
                    batch["features"], batch["session_indices"],
                    input_lengths=batch["input_lengths"],
                )
            # CRITICAL: cast OUTSIDE the autocast context, before the
            # loss -- see the module docstring.
            log_probs = log_probs.float()
        else:
            log_probs = model(
                batch["features"], batch["session_indices"],
                input_lengths=batch["input_lengths"],
            )

        ctc_lengths = model.ctc_input_lengths(batch["input_lengths"])
        loss = criterion(log_probs, batch["targets"], ctc_lengths, batch["target_lengths"])

        # day_layer_penalty added to the CTC loss BEFORE backward, so it
        # actually participates in the gradient -- see this function's
        # docstring entry. Gated on `model.day_layers is not None` since
        # there's nothing to regularize when day layers are disabled;
        # the DETECTION block just below deliberately still uses the
        # CTC-only `raw_per_item_loss` (via `log_probs`, unaffected by
        # this), since the penalty term is bounded and well-behaved by
        # construction and has no failure mode this check is for.
        if day_layer_penalty > 0 and getattr(model, "day_layers", None) is not None:
            loss = loss + day_layer_penalty * model.day_layers.identity_penalty()

        # DETECTION IS SEPARATE FROM THE REAL LOSS, ON PURPOSE. With
        # zero_infinity=True (as criterion is expected to be), an
        # infinite per-item loss is replaced with 0 BEFORE the batch is
        # ever reduced to the scalar `loss` above -- checking
        # `torch.isfinite(loss)` directly would essentially never fire,
        # since the very mechanism that makes training safe to continue
        # also erases the evidence that anything happened (verified
        # directly: a batch with an infeasible alignment returns exactly
        # 0.0, not inf, under zero_infinity=True). Detecting it instead
        # requires the RAW, per-item loss with zero_infinity=False and
        # reduction="none" -- computed here purely for detection (no_grad,
        # a second, cheap CTC forward-backward relative to the GRU's own
        # cost), never used as the training signal itself.
        with torch.no_grad():
            raw_per_item_loss = torch.nn.functional.ctc_loss(
                log_probs, batch["targets"], ctc_lengths, batch["target_lengths"],
                blank=getattr(criterion, "blank", 0), reduction="none", zero_infinity=False,
            )
        if not torch.isfinite(raw_per_item_loss).all():
            n_inf_batches += 1
            continue  # never step from a broken (inf/nan) gradient

        if scaler is not None:
            scaler.scale(loss).backward()
            # REQUIRED before clipping: without this, clip_grad_norm_
            # sees gradients still multiplied by the scale factor, so
            # grad_clip is compared against the wrong (scaled) magnitude
            # entirely -- the clip threshold becomes meaningless.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # scaler.step() silently skips optimizer.step() if the
            # UNSCALED gradient it just saw is inf/nan (routine during
            # AMP's first few steps, before GradScaler's dynamic loss
            # scale has calibrated down -- observed directly running
            # this project's own smoke test: grad_norm_mean was `inf` on
            # both smoke epochs with AMP on, finite without it, with
            # n_inf_batches=0 in both -- i.e. the CTC loss itself was
            # fine; the GRADIENT briefly overflowed under fp16, a
            # separate, AMP-specific transient this project's own
            # zero_infinity-based CTC check does not and should not
            # cover). scaler.step()/update() handle the skip-and-rescale
            # correctly on their own; the only thing this loop must not
            # do is silently let that batch's now-meaningless `inf`
            # grad_norm poison the epoch's average below.
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_trials += batch_size
        if math.isfinite(grad_norm):
            grad_norms.append(float(grad_norm))

        if loggers is not None:
            loggers.log_scalars(global_step_base + batch_idx, {"batch_loss": loss.item()})

    if total_trials == 0:
        warnings.warn(
            f"train_one_epoch: every batch this epoch ({n_batches}) had a "
            f"non-finite loss -- returning nan for 'loss' and "
            f"'grad_norm_mean'. This is a serious training-health signal, "
            f"not a normal occurrence.",
            stacklevel=2,
        )
        mean_loss = math.nan
        mean_grad_norm = math.nan
    else:
        mean_loss = total_loss / total_trials
        if grad_norms:
            mean_grad_norm = sum(grad_norms) / len(grad_norms)
        else:
            # Every stepped batch had a non-finite grad_norm (an AMP
            # overflow transient -- see the scaler.step() comment above).
            # The loss itself was fine (that's why we got here at all),
            # so 'loss' above is still meaningful; only grad_norm_mean
            # has nothing finite to average.
            warnings.warn(
                "train_one_epoch: every batch's gradient norm was "
                "non-finite this epoch (AMP overflow transient) -- "
                "returning nan for 'grad_norm_mean'. 'loss' is still "
                "meaningful.",
                stacklevel=2,
            )
            mean_grad_norm = math.nan

    return {
        "loss": mean_loss,
        "n_batches": n_batches,
        "n_inf_batches": n_inf_batches,
        "grad_norm_mean": mean_grad_norm,
        "seconds": time.perf_counter() - start,
    }


def train_one_step_accumulated(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
    day_layer_penalty: float = 0.0,
) -> dict[str, Any]:
    """One optimizer step, with its gradient ACCUMULATED across every
    microbatch `dataloader` yields -- mathematically identical to a single
    full-batch forward/backward over all of `dataloader`'s trials at once,
    but with peak activation memory bounded by the largest microbatch
    rather than by the total trial count.

    WHY THIS EXISTS (Phase 3, Arm 2/3a/3b). A "step" for these arms is
    defined as one full-batch gradient step over all N calibration trials
    (`docs/FINDINGS.md`: the N=100 recalibration-grid slowdown). At N=100,
    padding every trial to the longest (up to 2,355 bins across the
    recalibration grid's 7 sessions) and patching to
    7,168 dims produces activations large enough to push this project's
    8 GB GPU into WDDM's driver-level spill-to-system-RAM path (observed:
    ~2.47 s/step at N=100 vs. ~0.093 s/step at N=10 -- a 26x slowdown for
    10x the data). This function makes "one step over N trials" a matter
    of ACCUMULATING gradient over ceil(N / microbatch_size) forward/backward
    passes before ever calling `optimizer.step()` once -- same total
    gradient, same resulting update, bounded peak memory.

    THE MEAN-REDUCTION WEIGHTING PROBLEM, AND HOW IT'S HANDLED. `criterion`
    is expected to be an `nn.CTCLoss` with its default `reduction="mean"`:
    each microbatch's `criterion(...)` call returns `mean_{i in batch}
    (raw_loss_i / target_length_i)`, a MEAN over however many trials
    happen to be in THAT microbatch -- not over all `N` trials. Naively
    summing these per-microbatch means and calling `.backward()` on each
    would silently up-weight small (e.g. the last, partial) microbatches
    and NOT reproduce the true full-batch mean's gradient. The fix: scale
    each microbatch's loss by `(microbatch_size / N)` before calling
    `.backward()` on it (gradients accumulate additively across multiple
    `.backward()` calls with no intervening `optimizer.zero_grad()`, which
    is exactly what "accumulation" means here). Since
    `sum(microbatch_size / N for every microbatch) == 1` by construction
    (every trial appears in exactly one microbatch), this reconstructs
    precisely `grad(mean_{i in ALL N trials} (raw_loss_i / target_length_i))`
    -- the same gradient a single full-batch call to `criterion` would have
    produced. `tests/test_loop.py`'s accumulation-equivalence test verifies
    this numerically against a real full-batch call on the same data,
    rather than trusting the derivation alone.

    `day_layer_penalty * model.day_layers.identity_penalty()` does NOT
    depend on which trials are in the batch, only on the model's current
    (fixed, since no step happens mid-accumulation) parameters -- so unlike
    the CTC loss, it is added and backpropagated EXACTLY ONCE (after the
    microbatch loop, not scaled/repeated per microbatch), which already
    reproduces its exact full-step contribution.

    INFINITE-LOSS DETECTION IS ALL-OR-NOTHING FOR THE WHOLE STEP, not
    per-microbatch. If any microbatch's raw per-item loss is non-finite,
    accumulating the OTHER microbatches' gradients and stepping anyway
    would silently take a step from an incomplete gradient (the "full
    step" this function promises never actually happened) -- worse than
    `train_one_epoch`'s per-batch skip, since here "the batch" IS the
    step. Detected the same way as `train_one_epoch` (a raw,
    `zero_infinity=False`, `reduction="none"` CTC forward, purely for
    detection); on the first bad microbatch, every gradient accumulated
    so far this call is discarded (`optimizer.zero_grad()`) and the
    function returns immediately with `stepped=False`, without processing
    any remaining microbatches.

    Args:
        model, criterion, device: As in `train_one_epoch`.
        dataloader: Yields microbatches (e.g. `batch_size=16`) covering
            ALL trials this step should train on, in a fixed order
            (`shuffle=False` -- order doesn't matter for a full-batch-
            equivalent gradient, and a fixed order keeps this function
            deterministic given a deterministic model). `dataloader.dataset`
            must support `len()` (true of every `TrialDataset` in this
            project) -- its length is `N`, the weighting denominator.
        optimizer: Stepped exactly once per call (or zero times, if an
            infinite loss is detected) -- construct a FRESH optimizer per
            fit, same as every other Phase 3 arm (`DayLayers`' Adam
            stale-state caveat).
        grad_clip: Applied ONCE, after every microbatch's gradient has
            been accumulated (and after the day-layer penalty's) -- the
            same single clip a full-batch step would have applied, not
            one clip per microbatch.
        day_layer_penalty: Weight on `model.day_layers.identity_penalty()`,
            added once as described above. `0.0` (default): no penalty, a
            no-op identical to omitting it, matching `train_one_epoch`'s
            own default.

    Returns:
        A dict with keys:
          - `"loss"`: the full-batch-equivalent mean CTC loss (reconstructed
            via the same weighted-sum trick as the gradient) PLUS the
            day-layer penalty term, matching `train_one_epoch`'s
            convention that its reported `"loss"` includes the penalty.
            `nan` if the step was skipped (see `"stepped"` below).
          - `"n_microbatches"`: how many microbatches `dataloader` yielded.
          - `"n_inf_microbatches"`: `1` if the step was skipped due to a
            non-finite microbatch, else `0` (never more than 1: detection
            stops at the first bad microbatch rather than continuing to
            count).
          - `"grad_norm"`: `clip_grad_norm_`'s return value (the PRE-clip
            total gradient norm across every accumulated microbatch), or
            `nan` if the step was skipped.
          - `"stepped"`: `True` if `optimizer.step()` was called, `False`
            if the step was skipped (non-finite loss).
          - `"seconds"`: wall-clock time for this call.
    """
    if isinstance(criterion, torch.nn.CTCLoss) and not criterion.zero_infinity:
        warnings.warn(
            "train_one_step_accumulated: criterion is an nn.CTCLoss with "
            "zero_infinity=False. An infinite per-item loss will not be "
            "zeroed by criterion itself; this function's own detection "
            "pass still catches it and skips the step, but zero_infinity="
            "True is expected for consistency with train_one_epoch.",
            stacklevel=2,
        )

    model.train()
    start = time.perf_counter()

    total_trials = len(dataloader.dataset)
    if total_trials == 0:
        raise ValueError("train_one_step_accumulated: dataloader.dataset is empty.")

    optimizer.zero_grad()

    total_weighted_loss = 0.0
    n_microbatches = 0
    for raw_batch in dataloader:
        n_microbatches += 1
        batch = _move_batch_to_device(raw_batch, device)
        batch_size = batch["features"].shape[0]
        weight = batch_size / total_trials

        log_probs = model(
            batch["features"], batch["session_indices"], input_lengths=batch["input_lengths"],
        )
        ctc_lengths = model.ctc_input_lengths(batch["input_lengths"])

        # Detection pass -- see this function's docstring on why a bad
        # microbatch aborts the WHOLE step rather than being skipped alone.
        with torch.no_grad():
            raw_per_item_loss = torch.nn.functional.ctc_loss(
                log_probs, batch["targets"], ctc_lengths, batch["target_lengths"],
                blank=getattr(criterion, "blank", 0), reduction="none", zero_infinity=False,
            )
        if not torch.isfinite(raw_per_item_loss).all():
            optimizer.zero_grad()  # discard whatever prior microbatches accumulated
            return {
                "loss": math.nan, "n_microbatches": n_microbatches, "n_inf_microbatches": 1,
                "grad_norm": math.nan, "stepped": False, "seconds": time.perf_counter() - start,
            }

        loss = criterion(log_probs, batch["targets"], ctc_lengths, batch["target_lengths"])
        (loss * weight).backward()
        total_weighted_loss += loss.item() * weight

    if day_layer_penalty > 0 and getattr(model, "day_layers", None) is not None:
        penalty = day_layer_penalty * model.day_layers.identity_penalty()
        penalty.backward()
        total_weighted_loss += penalty.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    stepped = math.isfinite(grad_norm)
    if stepped:
        optimizer.step()

    return {
        "loss": total_weighted_loss,
        "n_microbatches": n_microbatches,
        "n_inf_microbatches": 0,
        "grad_norm": float(grad_norm) if stepped else math.nan,
        "stepped": stepped,
        "seconds": time.perf_counter() - start,
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    blank_index: int = 0,
    compute_per: bool = True,
) -> dict[str, Any]:
    """Run evaluation over a dataloader without updating model weights.

    Args:
        model: The model being evaluated.
        dataloader: Yields batches as produced by `collate_trials`.
        criterion: Loss function (expected to be `nn.CTCLoss`), computed
            the same way as `train_one_epoch` (fp32; no AMP here -- there
            is no forward-pass speed/backward-pass tradeoff to make when
            nothing is being backpropagated).
        device: Device to run computation on.
        blank_index: Passed to `greedy_ctc_decode` when `compute_per` is
            True.
        compute_per: If True (the default), greedy-decode every batch
            (`evaluation.ctc_decode.greedy_ctc_decode` -- note the axis
            convention: `model.forward` returns `(time, batch, classes)`,
            `greedy_ctc_decode` expects `(batch, time, classes)`, so the
            transpose here is mandatory, not cosmetic) and compute
            per-trial `EditCounts` against the true target sequence
            (`targets` split back into per-item sequences using
            `target_lengths` -- the same concatenation `collate_trials`
            produced them from). If False, skip decoding entirely (faster
            when only the loss is needed).

    Returns:
        A dict with keys:
          - `"loss"`: trial-count-weighted average CTC loss.
          - `"per"`: `evaluation.metrics.aggregate_error_rate` over every
            trial's `EditCounts`, or `None` if `compute_per=False`. A
            FRACTION (edits / reference phonemes), NOT a percentage --
            `0.35` means 35% PER, and values above `1.0` are legitimate
            (more edits than reference phonemes, e.g. from an
            undertrained model whose decode is mostly insertions -- see
            `evaluation.metrics.aggregate_error_rate`'s docstring). This
            convention is used consistently everywhere PER is computed,
            logged, or written by this module -- nothing here multiplies
            by 100.
          - `"counts"`: `list[EditCounts]`, one per trial, in dataloader
            order -- suitable input to
            `evaluation.metrics.bootstrap_error_rate_ci`. Empty if
            `compute_per=False`.
          - `"counts_by_session"`: `dict[str, list[EditCounts]]`, the
            same counts grouped by `session_id` -- suitable input to
            `evaluation.metrics.per_session_metrics`. Empty if
            `compute_per=False`.
          - `"n_trials"`: total trials evaluated.
          - `"seconds"`: wall-clock time for this call.
          - `"blank_fraction"`: fraction of VALID (non-padded) timesteps,
            pooled across every trial, whose argmax over the class axis
            is `blank_index` -- computed on the raw, pre-collapse
            `log_probs`, since this is a property of the frame-level
            output, not of the decoded sequence. Masked to `t <
            ctc_lengths[i]` per item `i`: `GRUCTC.forward` zeros the
            INPUT at padded timesteps, but the GRU still produces
            (meaningless) output there, so an unmasked count would be
            wrong, not just imprecise. Uses `ctc_lengths`
            (`model.ctc_input_lengths(...)`), not `batch["input_lengths"]`
            directly -- identical today (no temporal downsampling), but
            `ctc_lengths` is the one that actually matches `log_probs`'s
            time axis once downsampling is added (`models/gru_ctc.py`'s
            "THE CTC LENGTH CONTRACT"; `CLAUDE.md`'s "known pitfalls").
            An aggregate fraction (total blank frames / total valid
            frames), not a mean of per-batch fractions -- same reasoning
            as `evaluation.metrics`'s aggregate-vs-mean-of-rates argument
            (its module docstring). Expected to start near 1.0 (an
            undertrained CTC model emits almost all blanks) and fall as
            the model begins committing to phonemes -- the "blank phase"
            this diagnostic exists to detect the end of. `None` if
            `compute_per=False`.
          - `"decoded_to_reference_length_ratio"`: total decoded tokens
            (post-collapse, across every trial) divided by total
            reference phoneme count (across every trial) -- an aggregate
            ratio, not a mean of per-trial ratios, for the same reason as
            `"blank_fraction"` and `"per"` above. `None` if
            `compute_per=False`.
          - `"decoded_token_counts"`: `dict[int, int]`, class index ->
            number of times that class appears in the DECODED
            (post-collapse) output, summed over every trial. Contains
            only classes that appeared at least once. Empty dict (not
            `None`) if `compute_per=False`.
          - `"silence_deletions"` / `"phoneme_deletions"`: aggregate counts,
            across every trial, of deletion operations (from the same
            alignment `"counts"` is built from) whose deleted REFERENCE
            token is the silence class (`b2t_drift.phonemes.SILENCE_INDEX`)
            vs. any other (real-phoneme) class. `silence_deletions +
            phoneme_deletions` always equals the total deletion count
            summed over `"counts"`. Distinguishes "losing the
            silence/word-boundary marker specifically" from "channels
            going quiet on any phoneme" (`docs/DESIGN.md` section 4) --
            information `EditCounts`'s aggregate deletion count alone
            cannot carry. `None` if `compute_per=False`.
    """
    model.eval()
    start = time.perf_counter()

    total_loss = 0.0
    total_trials = 0
    counts: list[EditCounts] = []
    counts_by_session: dict[str, list[EditCounts]] = {}

    n_blank_frames = 0
    n_valid_frames = 0
    total_decoded_tokens = 0
    total_reference_tokens = 0
    decoded_token_counts: Counter[int] = Counter()
    silence_deletions = 0
    phoneme_deletions = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_batch_to_device(raw_batch, device)
            batch_size = batch["features"].shape[0]

            log_probs = model(
                batch["features"], batch["session_indices"],
                input_lengths=batch["input_lengths"],
            )
            ctc_lengths = model.ctc_input_lengths(batch["input_lengths"])
            loss = criterion(log_probs, batch["targets"], ctc_lengths, batch["target_lengths"])

            total_loss += loss.item() * batch_size
            total_trials += batch_size

            if compute_per:
                # blank_fraction: raw, pre-collapse log_probs (time, batch,
                # classes) -- see the docstring's "blank_fraction" entry
                # for why ctc_lengths (not batch["input_lengths"]) is the
                # correct mask here.
                time_indices = torch.arange(log_probs.size(0), device=log_probs.device)
                valid_mask = (
                    time_indices.unsqueeze(1) < ctc_lengths.to(log_probs.device).unsqueeze(0)
                )  # (time, batch)
                frame_argmax = log_probs.argmax(dim=-1)  # (time, batch)
                n_blank_frames += int(((frame_argmax == blank_index) & valid_mask).sum().item())
                n_valid_frames += int(valid_mask.sum().item())

                decoded = greedy_ctc_decode(
                    log_probs.transpose(0, 1),  # (time, batch, C) -> (batch, time, C)
                    lengths=ctc_lengths,
                    blank_index=blank_index,
                    n_classes=model.n_classes,
                )

                # Split the concatenated target sequence back into
                # per-item sequences using target_lengths -- the CPU
                # (never moved to device) copy from raw_batch, matching
                # how collate_trials concatenated them in the first place.
                target_lengths_list = raw_batch["target_lengths"].tolist()
                targets_list = raw_batch["targets"].tolist()
                offset = 0
                for i, length in enumerate(target_lengths_list):
                    true_sequence = targets_list[offset:offset + length]
                    offset += length

                    trial_counts, deleted_tokens = levenshtein_alignment_with_deleted_tokens(
                        true_sequence, decoded[i],
                    )
                    counts.append(trial_counts)
                    session_id = raw_batch["session_ids"][i]
                    counts_by_session.setdefault(session_id, []).append(trial_counts)

                    total_decoded_tokens += len(decoded[i])
                    total_reference_tokens += length
                    decoded_token_counts.update(decoded[i])
                    for tok in deleted_tokens:
                        if tok == SILENCE_INDEX:
                            silence_deletions += 1
                        else:
                            phoneme_deletions += 1

    mean_loss = total_loss / total_trials if total_trials > 0 else math.nan
    per = aggregate_error_rate(counts) if compute_per else None

    if compute_per:
        blank_fraction = n_blank_frames / n_valid_frames if n_valid_frames > 0 else math.nan
        if total_reference_tokens > 0:
            decoded_to_reference_length_ratio = total_decoded_tokens / total_reference_tokens
        else:
            # Matches EditCounts.error_rate / aggregate_error_rate's own
            # degenerate-denominator convention.
            decoded_to_reference_length_ratio = (
                0.0 if total_decoded_tokens == 0 else float("inf")
            )
    else:
        blank_fraction = None
        decoded_to_reference_length_ratio = None

    return {
        "loss": mean_loss,
        "per": per,
        "counts": counts,
        "counts_by_session": counts_by_session,
        "n_trials": total_trials,
        "seconds": time.perf_counter() - start,
        "blank_fraction": blank_fraction,
        "decoded_to_reference_length_ratio": decoded_to_reference_length_ratio,
        "decoded_token_counts": dict(decoded_token_counts) if compute_per else {},
        "silence_deletions": silence_deletions if compute_per else None,
        "phoneme_deletions": phoneme_deletions if compute_per else None,
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_epochs: int,
    patience: int,
    run_dir: Path,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip: float = 1.0,
    train_sampler: LengthBucketedSampler | None = None,
    loggers: list[RunLogger] | None = None,
    config: dict[str, Any] | None = None,
    day_layer_penalty: float = 0.0,
) -> dict[str, Any]:
    """Run the full train/eval loop with early stopping and checkpointing.

    EARLY STOPPING IS ON VALIDATION PER, NOT VALIDATION LOSS. PER is the
    quantity this project reports (`docs/DESIGN.md` section 9) -- CTC
    loss and PER can diverge (a model can keep lowering its loss on
    already-easy alignments while PER stalls or worsens on the harder
    ones), and stopping on loss would risk selecting a checkpoint that is
    NOT the best decoder by the metric that actually matters here. This
    is a deliberate choice, not the more common "stop on loss" default.

    `val_per` (here, in `metrics.csv`'s `val_per` column, and in every
    logger this function drives) is always a FRACTION, not a percentage
    -- see `evaluate`'s `"per"` docstring entry. A freshly-initialized or
    barely-trained model routinely reports `val_per > 1.0` (e.g. `3.59`
    means 359% PER, not 3.59%); this is expected, not a units bug.

    Args:
        model, train_loader, val_loader, optimizer, criterion, device:
            As in `train_one_epoch` / `evaluate`.
        max_epochs: Upper bound on epochs to run.
        patience: Epochs with no `val_per` improvement before stopping
            early. `val_per` at epoch 0 always counts as an improvement
            (there is no prior epoch to compare against), so training
            always runs at least 2 epochs before patience can trigger.
        run_dir: Directory for `checkpoint_best.pt`, `checkpoint_last.pt`,
            (via `training.logging.CSVLogger`) `metrics.csv`, and
            `decoded_token_counts.jsonl` (one JSON line per epoch,
            `{"epoch": ..., "counts": {"<class idx>": <count>, ...}}` --
            `evaluate`'s full `"decoded_token_counts"` histogram, kept out
            of `metrics.csv` itself since `CSVLogger` fixes its column set
            from the first call's keys and a variable-width histogram
            would either break that or force ~41 mostly-empty columns).
            Created if missing.
        scaler, grad_clip, train_sampler: Passed straight through to
            `train_one_epoch` each epoch (`train_sampler` as its
            `sampler` argument).
        scheduler: Optional LR scheduler, `.step()`-ed once per epoch,
            after both `train_one_epoch` and `evaluate`.
        loggers: Optional list of additional `RunLogger`s (e.g. a
            `TensorBoardLogger`, `WandbLogger`, `MLflowLogger`) layered
            ON TOP of the always-present `ConsoleLogger` + `CSVLogger`
            (`training.logging`'s module docstring: "the training loop
            degrades to Console + CSV if none are installed") for the
            once-per-EPOCH summary row -- `fit` constructs those two
            itself regardless of what's passed here, so `metrics.csv` is
            written even if `loggers=None`. These SAME `loggers` (but
            deliberately NOT `ConsoleLogger`/`CSVLogger`) are also handed
            to `train_one_epoch` for its once-per-BATCH `"batch_loss"`
            logging: `CSVLogger` is excluded there because its columns
            are fixed by whichever call establishes them first, and a
            per-batch `{"batch_loss": ...}` row has a different key set
            than the once-per-epoch summary row -- routing both through
            the same `CSVLogger` instance would raise on the second,
            differently-shaped call. `ConsoleLogger` is excluded from the
            per-batch stream too, simply to avoid printing one line per
            batch.
        config: The fully resolved experiment config for this run, saved
            into every checkpoint (`training.checkpoint.save_checkpoint`)
            so each one is self-describing. Defaults to `{}` if not
            given.
        day_layer_penalty: Passed straight through to `train_one_epoch`
            each epoch. Also controls three EXTRA `metrics.csv` columns,
            `day_penalty`, `day_layer_mean_max_weight_dev`,
            `day_layer_max_weight_dev` (`DayLayers.identity_penalty()`
            and `.weight_deviation_stats()`, read from `model.day_layers`
            AFTER each epoch's training, i.e. "value at the end of this
            epoch") -- these are logged whenever `model.day_layers is not
            None`, regardless of whether `day_layer_penalty > 0`, since
            they are a cheap read of existing state and useful to see
            drift away from identity even when nothing is actively
            pulling it back (e.g. the `A` baseline run in the Task 3
            comparison this exists for). `None` in all three columns when
            `model.day_layers is None`.

    Returns:
        A summary dict with keys `"best_epoch"`, `"best_val_per"`,
        `"final_epoch"` (the last epoch actually run, whether or not
        early stopping triggered), and `"total_seconds"`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = config or {}

    # Epoch-level: Console + CSV always, plus whatever extra trackers
    # were requested. Batch-level (handed to train_one_epoch): the SAME
    # extra trackers only -- never Console (too noisy) or CSV (its
    # columns are fixed by the first call's keys, and a per-batch row
    # has different keys than a per-epoch row) -- see this function's
    # `loggers` docstring entry for the full reasoning.
    epoch_loggers = MultiLogger([ConsoleLogger(), CSVLogger(run_dir), *(loggers or [])])
    batch_loggers = MultiLogger(loggers) if loggers else None

    best_val_per = float("inf")
    best_epoch: int | None = None
    epochs_since_improvement = 0
    final_epoch = -1
    fit_start = time.perf_counter()

    for epoch in range(max_epochs):
        final_epoch = epoch
        train_result = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, grad_clip=grad_clip, epoch=epoch,
            sampler=train_sampler, loggers=batch_loggers,
            day_layer_penalty=day_layer_penalty,
        )
        eval_result = evaluate(model, val_loader, criterion, device, compute_per=True)

        if scheduler is not None:
            scheduler.step()

        # Read AFTER this epoch's training, i.e. "day-layer state at the
        # end of epoch N" -- see fit's day_layer_penalty docstring entry
        # for why this is logged whenever day_layers exist, not only when
        # the penalty is actively applied.
        if getattr(model, "day_layers", None) is not None:
            day_penalty = model.day_layers.identity_penalty().item()
            day_layer_mean_max_weight_dev, day_layer_max_weight_dev = model.day_layers.weight_deviation_stats()
        else:
            day_penalty = None
            day_layer_mean_max_weight_dev = None
            day_layer_max_weight_dev = None

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": eval_result["loss"],
            "val_per": eval_result["per"],
            "grad_norm_mean": train_result["grad_norm_mean"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": train_result["seconds"] + eval_result["seconds"],
            "train_seconds": train_result["seconds"],
            "eval_seconds": eval_result["seconds"],
            "n_inf_batches": train_result["n_inf_batches"],
            "val_blank_fraction": eval_result["blank_fraction"],
            "val_decoded_length_ratio": eval_result["decoded_to_reference_length_ratio"],
            "val_n_distinct_decoded_tokens": len(eval_result["decoded_token_counts"]),
            "day_penalty": day_penalty,
            "day_layer_mean_max_weight_dev": day_layer_mean_max_weight_dev,
            "day_layer_max_weight_dev": day_layer_max_weight_dev,
        }
        epoch_loggers.log_scalars(epoch, row)

        # Full per-class histogram, kept out of metrics.csv (see run_dir's
        # docstring entry above) -- one JSON line per epoch.
        with (run_dir / "decoded_token_counts.jsonl").open("a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "counts": {str(k): v for k, v in eval_result["decoded_token_counts"].items()},
            }) + "\n")

        checkpoint_metrics = {
            "val_per": eval_result["per"],
            "val_loss": eval_result["loss"],
            "train_loss": train_result["loss"],
            "val_blank_fraction": eval_result["blank_fraction"],
            "val_decoded_length_ratio": eval_result["decoded_to_reference_length_ratio"],
            "val_n_distinct_decoded_tokens": len(eval_result["decoded_token_counts"]),
            "day_penalty": day_penalty,
            "day_layer_mean_max_weight_dev": day_layer_mean_max_weight_dev,
            "day_layer_max_weight_dev": day_layer_max_weight_dev,
        }
        save_checkpoint(
            run_dir / "checkpoint_last.pt", model, optimizer, epoch,
            checkpoint_metrics, config,
        )

        improved = eval_result["per"] < best_val_per
        if improved:
            best_val_per = eval_result["per"]
            best_epoch = epoch
            epochs_since_improvement = 0
            save_checkpoint(
                run_dir / "checkpoint_best.pt", model, optimizer, epoch,
                checkpoint_metrics, config,
            )
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(
                f"fit: stopping early at epoch {epoch} "
                f"(no val_per improvement for {patience} epochs)."
            )
            break

    if best_epoch is not None:
        epoch_loggers.log_artifact(run_dir / "checkpoint_best.pt")
    epoch_loggers.finish()

    return {
        "best_epoch": best_epoch,
        "best_val_per": best_val_per,
        "final_epoch": final_epoch,
        "total_seconds": time.perf_counter() - fit_start,
    }
