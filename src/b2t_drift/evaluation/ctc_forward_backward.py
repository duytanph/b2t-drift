"""CTC forward-backward: per-timestep posterior probabilities over classes.

Responsible for the standard CTC alpha/beta recursion (Graves 2006), computed
in log space, given a model's emissions (`log_probs`) and the KNOWN target
label sequence. This is fundamentally different from `ctc_decode.py`'s
greedy decoding: decoding commits to the single argmax path and discards
everything else, while forward-backward marginalizes over EVERY valid
alignment consistent with the target, producing a full posterior
distribution over classes at each timestep. Phase 2's tuning-matrix analysis
needs the posterior specifically because a bin is rarely purely one phoneme
-- "0.6 probability of /K/, 0.4 of blank" should contribute 0.6 to /K/'s
running average, not be hard-assigned to whichever class happens to be
largest (`docs/DESIGN.md` section 4's attribution question).

THE EXTENDED-SEQUENCE CONSTRUCTION. For a target `y = (y_1, ..., y_L)` (no
blanks), CTC's alignment lattice is built over the EXTENDED sequence
`z = (blank, y_1, blank, y_2, blank, ..., y_L, blank)`, length `S = 2L + 1`
-- a blank before, after, and between every label, so the lattice can
represent "stay on this label," "advance to a blank," or "advance to the
next distinct label" at every step, and repeated-label targets (e.g. two
consecutive identical phonemes) are only decodable with an intervening
blank to prevent CTC's repeat-collapsing rule from merging them.

THE BETA-VIA-TIME-REVERSAL TRICK. Computing beta with its own separate
recursion would duplicate the alpha code with all the indices mirrored --
easy to get subtly wrong. Instead: reversing BOTH the time axis of
`log_probs` and the extended sequence `z` produces an equivalent CTC
problem (the reversed extended sequence is `reverse(z) = (blank, y_L, ...,
blank, y_1, blank)`, which is exactly the correct extended sequence for the
REVERSED target `reverse(y)`). Running the SAME alpha recursion on this
reversed problem and reversing the result back gives beta directly:
`beta[t, s] = alpha_reversed[T-1-t, S-1-s]`. One recursion implementation,
used twice.

NUMERICAL PRECISION. The recursion runs in `float64` internally regardless
of the input `log_probs`' dtype -- summing `log_probs` across a session's
full time axis (patched length up to ~540) in `float32` risks accumulated
rounding error large enough to matter for the nn.CTCLoss-equivalence check
this module's own tests run against; `float64` throughout the DP keeps that
check meaningful to a tight tolerance.
"""

from __future__ import annotations

import torch

NEG_INF = float("-inf")


def _build_extended_sequence(target: torch.Tensor, blank: int) -> torch.Tensor:
    """`z = (blank, y_1, blank, y_2, ..., y_L, blank)`, length `2*L + 1`.

    Args:
        target: `(L,)` int64, the label sequence (no blanks, `L >= 0`).
        blank: The blank class index.

    Returns:
        `(2*L + 1,)` int64.
    """
    length = target.shape[0]
    device = target.device
    z = torch.full((2 * length + 1,), blank, dtype=torch.int64, device=device)
    z[1::2] = target
    return z


def _forward_log_alpha(log_probs: torch.Tensor, z: torch.Tensor, blank: int) -> torch.Tensor:
    """The alpha recursion, in log space, `float64` throughout.

    Args:
        log_probs: `(T, C)` float64 log-probabilities.
        z: `(S,)` int64 extended target sequence.
        blank: Blank class index.

    Returns:
        `(T, S)` float64 log-alpha.
    """
    t_steps, _ = log_probs.shape
    s_states = z.shape[0]
    device = log_probs.device

    # skip_allowed[s]: whether the "skip a blank" transition alpha[t-1,s-2]
    # -> alpha[t,s] is legal -- only when s >= 2, z[s] is not itself a
    # blank, and z[s] != z[s-2] (two IDENTICAL consecutive labels in the
    # target must have a blank between them; skipping would let CTC's
    # repeat-collapse rule silently merge them into one).
    skip_allowed = torch.zeros(s_states, dtype=torch.bool, device=device)
    if s_states >= 3:
        skip_allowed[2:] = (z[2:] != blank) & (z[2:] != z[:-2])

    log_alpha = torch.full((t_steps, s_states), NEG_INF, dtype=torch.float64, device=device)
    log_alpha[0, 0] = log_probs[0, z[0]]
    if s_states > 1:
        log_alpha[0, 1] = log_probs[0, z[1]]

    for t in range(1, t_steps):
        prev = log_alpha[t - 1]

        stay_or_advance = prev.clone()
        shifted1 = torch.cat([prev.new_full((1,), NEG_INF), prev[:-1]])
        stay_or_advance = torch.logaddexp(stay_or_advance, shifted1)

        if s_states >= 3:
            shifted2 = torch.cat([prev.new_full((2,), NEG_INF), prev[:-2]])
            shifted2 = torch.where(skip_allowed, shifted2, prev.new_full((), NEG_INF))
            stay_or_advance = torch.logaddexp(stay_or_advance, shifted2)

        log_alpha[t] = stay_or_advance + log_probs[t, z]

    return log_alpha


def ctc_forward_backward(
    log_probs: torch.Tensor,
    target: torch.Tensor,
    blank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CTC forward-backward for ONE sequence: per-timestep class posteriors
    and the total log-likelihood.

    Args:
        log_probs: `(T, C)` float, log-probabilities (already
            `log_softmax`-ed over the class dimension) for one trial, at
            whatever time resolution the model actually emits (e.g. the
            PATCHED resolution for `PretrainedBaselineRNN` -- see
            `map_posteriors_to_bins` for converting back to 20 ms bins).
        target: `(L,)` int64, the true label sequence, values in
            `[1, C-1]` (no blanks -- `L` may be 0, an all-blank/all-silence
            edge case handled correctly by the `S=1` extended sequence).
        blank: Blank class index (0 throughout this project,
            `b2t_drift.phonemes.BLANK_INDEX`).

    Returns:
        `(posteriors, log_likelihood)`:
          - `posteriors`: `(T, C)` float64. `posteriors[t]` sums to 1 (up
            to floating-point error) for every `t` -- a genuine posterior
            distribution over classes at that timestep, marginalized over
            every alignment consistent with `target`.
          - `log_likelihood`: 0-dim float64 tensor, `log p(target |
            log_probs)` -- matches `-nn.functional.ctc_loss(..., reduction
            ="none")` for the same inputs to numerical precision (this
            equivalence is this module's primary correctness check, since
            it is independent of this implementation's own internals --
            see `tests/test_ctc_forward_backward.py`).

    Raises:
        ValueError: If `T < L` (fewer timesteps than the RAW target
            length) -- CTC's hard minimum requirement (matching
            `data/collate.py`'s own "input length >= target length"
            contract elsewhere in this project); no valid alignment can
            exist below it. NOTE: `T < S` (the longer, 2L+1 extended
            length) is NOT checked here and is not by itself infeasible --
            a target with no consecutive repeated labels can be fully
            decoded using only "skip the blank" transitions, needing as
            few as `L` timesteps, not `S` (see this module's own test
            `test_forced_alignment_when_t_equals_target_length_no_repeats`).
            A target with forced consecutive-repeat blanks needs strictly
            more than `L`, up to `S`; this function does not compute that
            tighter bound and will return `log_likelihood = -inf` (not
            raise) if `T` turns out insufficient for those repeats --
            matching `nn.CTCLoss`'s own behavior for the same input.
    """
    log_probs64 = log_probs.to(torch.float64)
    t_steps = log_probs64.shape[0]
    z = _build_extended_sequence(target, blank)
    s_states = z.shape[0]
    target_length = target.shape[0]

    if t_steps < target_length:
        raise ValueError(
            f"ctc_forward_backward: T={t_steps} < target length "
            f"L={target_length} -- CTC requires input length >= target "
            f"length; no valid alignment can exist."
        )

    log_alpha = _forward_log_alpha(log_probs64, z, blank)

    # Beta via the time-and-sequence-reversal trick (see module docstring).
    reversed_log_probs = log_probs64.flip(0)
    reversed_z = z.flip(0)
    log_alpha_reversed = _forward_log_alpha(reversed_log_probs, reversed_z, blank)
    log_beta = log_alpha_reversed.flip((0, 1))

    log_z_total = torch.logaddexp(log_alpha[-1, -1], log_alpha[-1, -2] if s_states > 1 else log_alpha.new_tensor(NEG_INF))

    log_gamma = log_alpha + log_beta - log_probs64[torch.arange(t_steps).unsqueeze(1), z.unsqueeze(0)] - log_z_total
    # log_gamma[t, s] is currently log P(state=s at time t | obs, target)
    # computed as alpha*beta/(p(z_s at t)*Z) -- see the derivation note
    # below for why the log_probs term must be divided out.
    gamma = torch.exp(log_gamma)

    n_classes = log_probs.shape[1]
    posteriors = torch.zeros((t_steps, n_classes), dtype=torch.float64, device=log_probs.device)
    posteriors.scatter_add_(1, z.unsqueeze(0).expand(t_steps, -1), gamma)

    return posteriors, log_z_total


def map_posteriors_to_bins(posteriors: torch.Tensor, patch_stride: int) -> torch.Tensor:
    """Map PATCHED-resolution posteriors back down to 20 ms raw bins.

    `PretrainedBaselineRNN` (`models/pretrained_baseline.py`) emits one
    frame per `patch_stride` raw input bins (patch 14, stride 4 -- 80 ms
    output stride over 20 ms input bins), so `ctc_forward_backward`'s
    posteriors are indexed by PATCH, not by raw bin. Phase 2's tuning
    matrices are a posterior-weighted average of the RAW `input_features`
    vector at each bin, so the two time axes must agree.

    THE CHOICE MADE HERE: repeat each patched frame's posterior across the
    `patch_stride` raw bins it steps over, i.e. frame `f`'s distribution is
    assigned to raw bins `[f * patch_stride, (f+1) * patch_stride)`. This is
    one of the two options the task names (repeat-across-stride vs.
    assign-to-center-bin); repeat-across-stride was chosen because
    `patch_stride` (not `patch_size`) is exactly the number of NEW raw bins
    each successive frame advances by, so tiling by `patch_stride` covers
    every raw bin in `[0, n_patches * patch_stride)` EXACTLY ONCE, with no
    gaps and no double-assignment -- assigning to a single "center" bin
    per frame would instead leave most raw bins with no posterior at all
    (`patch_size=14 > patch_stride=4`, so consecutive frames' windows
    overlap; only 1 of every 4 raw bins would be a center bin under that
    scheme). The trailing `n_time_steps - n_patches * patch_stride` raw
    bins (at most `patch_size - patch_stride` of them -- the tail of the
    last patch's window that never starts a new frame) are NOT covered by
    this function's output and must be excluded by the caller, not
    silently assigned a stale posterior.

    Args:
        posteriors: `(T_patched, C)` float, from `ctc_forward_backward`.
        patch_stride: The model's `patch_stride` (4 for
            `PretrainedBaselineRNN`).

    Returns:
        `(T_patched * patch_stride, C)`, each patched row repeated
        `patch_stride` times along the time axis.
    """
    return posteriors.repeat_interleave(patch_stride, dim=0)
