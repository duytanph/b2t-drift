"""Posterior-weighted phoneme tuning matrices, and correlating them across
sessions.

A "tuning matrix" for one session is `(n_classes, n_features)`: row `c` is
the posterior-weighted mean 512-dim feature vector across every bin of every
labeled trial in that session, where the weight is `P(class=c | trial,
model)` at that bin (`evaluation.ctc_forward_backward`'s posteriors) --
NOT a hard per-bin class assignment (`docs/DESIGN.md` section 4's
tuning-change hypothesis: does the phoneme-to-channel mapping shift over
time?). Comparing two sessions' tuning matrices is therefore a direct probe
of whether a given phoneme's neural signature has moved.

WHY POSTERIOR-WEIGHTED, NOT HARD-ASSIGNED. A bin with 0.6 probability of
`/K/` and 0.4 of blank should contribute 0.6 of its feature vector to
`/K/`'s running average and 0.4 to blank's -- not all of it to whichever is
larger. Hard assignment would systematically bias the tuning matrix toward
whatever the model is MOST confident about, discarding exactly the
information (partial, ambiguous posterior mass) that a well-calibrated
forward-backward pass provides over greedy argmax decoding.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TuningMatrixAccumulator:
    """Accumulates posterior-weighted feature sums for one session, one
    trial at a time, without ever materializing the full per-bin data for
    the whole session at once.

    Two running sums are all that's needed: `sum_c(posterior * feature)`
    and `sum_c(posterior)` (the total "soft bin count" assigned to class
    `c`) -- the final tuning matrix is just the elementwise ratio.
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        """
        Args:
            n_features: Feature vector width (512 for the full dataset, or
                256 for a TX-only/SBP-only subset -- this class has no
                opinion about which; the caller's `add_trial` calls decide).
            n_classes: Number of classes (41: CTC blank + 39 phonemes +
                silence). Blank (class 0) accumulates like any other class
                here -- callers building a "40 non-blank classes" report
                simply skip row 0 when reading `matrix()`'s output; nothing
                about this class special-cases it.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.sum_features = np.zeros((n_classes, n_features), dtype=np.float64)
        self.sum_weights = np.zeros(n_classes, dtype=np.float64)

    def add_trial(self, posteriors: np.ndarray, features: np.ndarray) -> None:
        """Accumulate one trial's contribution.

        Args:
            posteriors: `(T, n_classes)`, e.g.
                `ctc_forward_backward.map_posteriors_to_bins`'s output --
                already at the SAME time resolution as `features`.
            features: `(T, n_features)`, the raw (or feature-subsetted)
                input features for the SAME `T` bins, bin-for-bin aligned
                with `posteriors`.

        Raises:
            ValueError: If `posteriors` and `features` disagree on `T`
                (the caller must have already truncated both to their
                common covered range -- e.g.
                `map_posteriors_to_bins`'s output covers
                `n_patches * patch_stride` raw bins, which is usually
                shorter than the trial's full `n_time_steps`), or if
                either's other dimension doesn't match this accumulator's
                `n_classes`/`n_features`.
        """
        posteriors = np.asarray(posteriors, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)
        if posteriors.shape[0] != features.shape[0]:
            raise ValueError(
                f"TuningMatrixAccumulator.add_trial: posteriors has "
                f"{posteriors.shape[0]} timesteps but features has "
                f"{features.shape[0]} -- they must be bin-for-bin aligned "
                f"(see map_posteriors_to_bins)."
            )
        if posteriors.shape[1] != self.n_classes:
            raise ValueError(
                f"TuningMatrixAccumulator.add_trial: posteriors has "
                f"{posteriors.shape[1]} classes, expected {self.n_classes}."
            )
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"TuningMatrixAccumulator.add_trial: features has "
                f"{features.shape[1]} columns, expected {self.n_features}."
            )

        self.sum_features += posteriors.T @ features
        self.sum_weights += posteriors.sum(axis=0)

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """The current tuning matrix and per-class total soft bin count.

        Returns:
            `(matrix, weights)`: `matrix` is `(n_classes, n_features)`,
            row `c` a NaN row if `weights[c] == 0` (never divided by zero
            silently -- `np.errstate` suppresses the RuntimeWarning since
            NaN is the intended, self-documenting result here, not an
            error). `weights` is `(n_classes,)`, the total posterior mass
            (summed over every bin of every trial passed to `add_trial`)
            assigned to each class -- report this alongside the matrix so
            a caller can flag any class with too little support to trust
            (see this module's docstring / Phase 2 Task B(b)).
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = self.sum_features / self.sum_weights[:, None]
        return matrix, self.sum_weights.copy()


def per_phoneme_correlation(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    class_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Mean Pearson correlation of each class's tuning ROW between two
    sessions' matrices -- NOT the flattened-matrix correlation.

    WHY PER-PHONEME MEAN, NOT FLATTENED. Flattening both matrices to one
    long vector and correlating conflates two different things: (1) the
    ACROSS-PHONEME pattern (phoneme A's overall feature profile differing
    from phoneme B's, which is largely a property of the phonemes
    themselves, not of drift) and (2) the WITHIN-PHONEME cross-channel
    structure this analysis actually wants to track (has /K/'s own
    channel signature moved?). A flattened correlation would stay high
    just because "the phonemes are still recognizably different from each
    other," even if every individual phoneme's channel profile had
    drifted substantially, provided the phonemes remained as different
    from EACH OTHER as before. Per-phoneme correlation isolates exactly
    the within-phoneme question by construction: each row is correlated
    only against its own same-class counterpart in the other session.

    Args:
        matrix_a, matrix_b: `(n_classes, n_features)`, same shape, same
            class indexing -- e.g. two sessions' `TuningMatrixAccumulator
            .matrix()` outputs.
        class_indices: Which row indices to include (e.g. `range(1, 41)`
            for "the 40 non-blank classes," or a healthy-channel-only
            COLUMN restriction applied by the caller before calling this
            function -- this function itself has no channel-health logic).
            Defaults to every row.

    Returns:
        A dict with `"mean_correlation"` (float, NaN if no class had a
        computable correlation), `"per_class"` (dict class_index ->
        Pearson r, `NaN` for a class whose row in EITHER matrix is
        constant (undefined correlation) or all-NaN (an unsupported class
        -- see `TuningMatrixAccumulator.matrix`'s NaN-row behavior)), and
        `"n_valid_classes"` (how many classes had a finite correlation,
        i.e. contributed to `"mean_correlation"`).
    """
    if class_indices is None:
        class_indices = list(range(matrix_a.shape[0]))

    per_class: dict[int, float] = {}
    valid_values: list[float] = []
    for c in class_indices:
        row_a = matrix_a[c]
        row_b = matrix_b[c]
        if np.any(np.isnan(row_a)) or np.any(np.isnan(row_b)):
            per_class[c] = float("nan")
            continue
        if np.std(row_a) == 0 or np.std(row_b) == 0:
            per_class[c] = float("nan")
            continue
        r = float(np.corrcoef(row_a, row_b)[0, 1])
        per_class[c] = r
        valid_values.append(r)

    mean_corr = float(np.mean(valid_values)) if valid_values else float("nan")
    return {
        "mean_correlation": mean_corr,
        "per_class": per_class,
        "n_valid_classes": len(valid_values),
    }


def flattened_matrix_correlation(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    class_indices: list[int] | None = None,
) -> float:
    """The alternative this module does NOT use as the primary measure
    (see `per_phoneme_correlation`'s docstring for why) -- kept as a
    cross-check, since the task allows either and asks which was chosen.

    Args:
        matrix_a, matrix_b: `(n_classes, n_features)`.
        class_indices: Which rows to include before flattening; defaults
            to every row.

    Returns:
        Pearson correlation of the two matrices' flattened (and any
        all-NaN rows excluded) values, or NaN if fewer than 2 finite
        pairs remain.
    """
    if class_indices is None:
        class_indices = list(range(matrix_a.shape[0]))
    a = matrix_a[class_indices].ravel()
    b = matrix_b[class_indices].ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2 or np.std(a[finite]) == 0 or np.std(b[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(a[finite], b[finite])[0, 1])
