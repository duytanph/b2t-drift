"""Raw firing-rate recovery from the T15 dataset's normalized features.

`input_features` is already rolling-per-sentence z-scored before release
(`docs/DATASET.md` section 9 item 7) -- there is no raw spike count left in
the shipped data. But because normalization is per (sentence, channel) --
one fixed `(sigma, mu)` affine map for the whole trial, for a given TX
channel -- the released, continuous-looking float values for a genuine
integer spike-count channel still sit on an evenly-spaced LATTICE within
one trial: consecutive integer counts map to evenly-spaced released values,
spacing `1/sigma` apart. This module recovers that lattice per
(trial, channel) and inverts it back to a raw count, then a firing rate.

THE EXACT ROUTE, NOT THE POISSON APPROXIMATION (`docs/DATASET.md` section 9
item 9(d)-(e); `CLAUDE.md`'s "known pitfalls" analog for this domain):

    sigma = 1 / (lattice spacing for this trial and channel)
    mu    = -(minimum observed value in this trial/channel) * sigma
    raw   = released * sigma + mu

`mu`'s minimum-value assumption is justified empirically (the minimum
recurs as a plateau across many trials, not a one-off extreme -- see the
FINDINGS.md entry this module implements), not re-derived here. The
Poisson approximation (`mu = minimum**2`) disagrees by ~1.5x and is
conceptually unsound (it treats a single pooled minimum as if one
`(sigma, mu)` applied across many sentences' different rolling-window
divisors, which items 7 and 9(c) of `docs/DATASET.md` section 9 rule out).
This module implements ONLY the exact per-(trial, channel) route.

WHY PER-TRIAL, NOT PER-BLOCK OR PER-SESSION. Normalization is a rolling
window over up to the previous 20 sentences, recomputed every sentence
(`docs/DATASET.md` section 9 item 7) -- there is no single shared divisor
across a block or session, so `recover_lattice_params` must be called once
per (trial, channel), never pooled across trials first.

SBP CHANNELS DO NOT LATTICE. Spike band power is continuous power, not an
integer count, so `recover_lattice_params` is expected to (and, per
`docs/FINDINGS.md`'s control check, empirically does) FAIL on SBP columns
-- this module has no channel-type awareness of its own; callers restrict
it to TX columns (0-255, `docs/DATASET.md` section 7) themselves.

LATTICE DETECTION FAILS SOMETIMES, EVEN ON GENUINE TX CHANNELS (previously
measured: 230/240, 95.8%) -- e.g. a trial too short to populate enough
distinct integer counts, or a rolling window straddling a real change in
firing statistics that breaks the single-affine-map assumption for that
one trial. `LatticeResult.success` and `.reason` exist so a caller MUST
see and explicitly handle a failure (count it, skip that one
(trial, channel) pair) rather than let a wrong `sigma`/`mu` silently
produce a fabricated firing rate -- `CLAUDE.md`'s validation philosophy
applied to this domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_WIDTH_SECONDS = 0.02  # 20 ms bins (docs/DATASET.md section 5)
DEFAULT_RELATIVE_TOLERANCE = 1e-3
# Need at least 2 gaps between unique values to have anything to validate
# a candidate lattice spacing against.
MIN_UNIQUE_VALUES = 3


@dataclass(frozen=True)
class LatticeResult:
    """Outcome of one (trial, channel) lattice-recovery attempt.

    Attributes:
        success: Whether a consistent lattice spacing was found.
        sigma: `1 / spacing`, or `None` if `success` is False.
        mu: The affine offset (see the module docstring's formula), or
            `None` if `success` is False.
        n_unique: Number of distinct values seen -- kept even on failure,
            for diagnosing WHY a (trial, channel) pair failed (e.g. too
            few distinct values vs. a genuinely non-lattice channel).
        reason: `None` on success; a short human-readable string naming
            the failure mode otherwise.
    """

    success: bool
    sigma: float | None
    mu: float | None
    n_unique: int
    reason: str | None = None


def recover_lattice_params(
    values: np.ndarray,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> LatticeResult:
    """Recover one trial's one channel's affine lattice map, if one exists.

    Method: take the distinct values `values` contains, sort them, and
    look at the gaps between consecutive distinct values. If this column
    is a z-scored integer count, every gap should be an (unknown) integer
    multiple of one common spacing (the count-to-released-value scale
    factor for this trial). The smallest observed gap is taken as a first
    candidate spacing; every gap is then checked for being close (within
    `relative_tolerance`) to an integer multiple of a REFINED spacing
    (the median of `gap / round(gap / candidate)` over all gaps, which
    averages out float rounding noise rather than trusting the single
    smallest gap exactly).

    Args:
        values: 1-D array of released (already z-scored) values for one
            trial's one channel, e.g. `features[:, channel]` for one
            trial. Any length, but see `MIN_UNIQUE_VALUES`.
        relative_tolerance: Maximum allowed relative deviation (fraction
            of the refined spacing) between any gap and its nearest
            integer multiple of that spacing, for this to count as a
            successful lattice recovery.

    Returns:
        A `LatticeResult`. `success=False` (with `reason` set) if: fewer
        than `MIN_UNIQUE_VALUES` distinct values are present; the smallest
        gap is non-positive (shouldn't happen after `np.unique`, guarded
        anyway); the refined spacing is non-positive; or any gap's
        relative deviation from its nearest integer multiple of the
        refined spacing exceeds `relative_tolerance`.
    """
    values = np.asarray(values, dtype=np.float64)
    unique_vals = np.unique(values)
    if unique_vals.size < MIN_UNIQUE_VALUES:
        return LatticeResult(
            False, None, None, int(unique_vals.size),
            reason=f"only {unique_vals.size} distinct value(s), need >= {MIN_UNIQUE_VALUES}",
        )

    gaps = np.diff(unique_vals)
    candidate_spacing = float(gaps.min())
    if candidate_spacing <= 0:
        return LatticeResult(
            False, None, None, int(unique_vals.size),
            reason="non-positive candidate spacing (should be unreachable after np.unique)",
        )

    multiples = np.round(gaps / candidate_spacing)
    multiples[multiples < 1] = 1  # guard against float division producing 0
    refined_spacing = float(np.median(gaps / multiples))
    if refined_spacing <= 0:
        return LatticeResult(
            False, None, None, int(unique_vals.size),
            reason="non-positive refined spacing",
        )

    relative_residuals = np.abs(gaps / multiples - refined_spacing) / refined_spacing
    max_residual = float(relative_residuals.max())
    if max_residual >= relative_tolerance:
        return LatticeResult(
            False, None, None, int(unique_vals.size),
            reason=f"max relative residual {max_residual:.4g} >= tolerance {relative_tolerance:.4g}",
        )

    sigma = 1.0 / refined_spacing
    mu = -float(unique_vals[0]) * sigma
    return LatticeResult(True, sigma, mu, int(unique_vals.size), reason=None)


def recover_raw_counts(values: np.ndarray, lattice: LatticeResult) -> np.ndarray:
    """Apply a successful `LatticeResult`'s affine map to recover raw counts.

    Args:
        values: The same array (or a same-trial-and-channel slice/superset
            of it) `recover_lattice_params` was called on.
        lattice: A `LatticeResult` with `success=True`.

    Returns:
        `values * lattice.sigma + lattice.mu`, the recovered raw spike
        counts (not rounded to integers -- callers needing integers should
        round explicitly, but a mean/rate computation has no need to).

    Raises:
        ValueError: If `lattice.success` is False -- there is no
            `sigma`/`mu` to apply. Callers must check `.success` (and
            handle the failure -- count it, skip this pair) before
            calling this, never assume success.
    """
    if not lattice.success:
        raise ValueError(
            "recover_raw_counts: lattice recovery did not succeed "
            f"(reason: {lattice.reason!r}) -- there is no sigma/mu to apply."
        )
    return values * lattice.sigma + lattice.mu


def trial_channel_firing_rate_hz(
    values: np.ndarray,
    bin_width_seconds: float = BIN_WIDTH_SECONDS,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> tuple[float | None, LatticeResult]:
    """Recover one trial's one TX channel's firing rate in Hz.

    Combines `recover_lattice_params` + `recover_raw_counts` + averaging
    over the trial's bins, converting from counts/bin to counts/second.

    Args:
        values: 1-D array, one trial's one TX channel's released values
            (e.g. `features[:, channel]`, already sliced to the trial's
            true, unpadded length).
        bin_width_seconds: Bin duration -- 0.02 (20 ms,
            `docs/DATASET.md` section 5) by default.
        relative_tolerance: Passed to `recover_lattice_params`.

    Returns:
        `(firing_rate_hz, lattice)`. `firing_rate_hz` is `None` when
        `lattice.success` is False -- callers MUST handle this explicitly
        (exclude this (trial, channel) pair and count the failure; do not
        substitute a default rate, which would silently fabricate data).
    """
    lattice = recover_lattice_params(values, relative_tolerance=relative_tolerance)
    if not lattice.success:
        return None, lattice
    raw_counts = recover_raw_counts(values, lattice)
    firing_rate_hz = float(np.mean(raw_counts)) / bin_width_seconds
    return firing_rate_hz, lattice
