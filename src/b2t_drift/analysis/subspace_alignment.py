"""Unsupervised feature-distribution alignment: CORAL and orthogonal
Procrustes (Phase 3, Arm 1a/1b -- docs/DESIGN.md section 5).

Both arms use ONLY the unlabeled features of a target session's calibration
trials (transcripts discarded) to correct for a distribution shift relative
to the training block's features, then apply the resulting linear map to
that session's evaluation trials before the UNMODIFIED (day-0) decoder.
Both reduce to: estimate a source covariance and a target covariance, then
solve for a linear map between them in closed form -- no gradient descent,
no labels.

  CORAL (`coral_alignment_matrix`): `A = C_target^(-1/2) @ C_source^(1/2)`
      -- whitens the target distribution then re-colors it with the
      source's covariance structure. Can express an arbitrary (not
      necessarily orthogonal) linear map: scaling and shearing along with
      rotation.
  Orthogonal Procrustes (`procrustes_alignment_matrix`) -- the classical
      Schonemann (1966) solution for the best ORTHOGONAL map between two
      matrices in the least-squares sense, applied to the two covariances'
      matrix square roots (themselves symmetric, so `sqrt_target.T ==
      sqrt_target`): find orthogonal `R` minimizing
      `||sqrt_target @ R - sqrt_source||_F`, given by `R = U @ Vt` from the
      SVD of `sqrt_target.T @ sqrt_source`. `R` can only rotate the target
      distribution's principal axes to align with the source's; it cannot
      rescale or shear, which is exactly the "more constrained than CORAL"
      property that should help when the calibration set is small and
      `C_target` itself is only loosely estimated -- an ill-conditioned
      target covariance corrupts CORAL's whitening step directly (dividing
      by near-zero eigenvalues), whereas Procrustes only ever asks "which
      way is this axis pointing," a much less noise-sensitive question.

Both matrix square roots go through the SAME `_symmetric_matrix_power`
helper (via eigendecomposition, since every covariance here is symmetric
PSD by construction), so a ridge term or shrinkage coefficient applied to
one is applied identically to the other -- there is exactly one place
either implementation could get the numerics wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RIDGE_EPS = 1e-6


class CovarianceAccumulator:
    """Streams raw `(n_timesteps, n_features)` feature blocks into a running
    mean and (biased) covariance, without ever holding every trial's
    features in memory at once.

    Matches this project's "lazy loading" discipline (`data/dataset.py`'s
    module docstring) -- computing `C_source` from the s05 training block's
    908 trials should not require materializing all of their concatenated
    timesteps as one array, since a handful of long trials at up to ~1,735
    timesteps each would otherwise force an avoidable, single large
    allocation.

    Uses the standard streaming identity `Cov = E[xx^T] - E[x]E[x]^T`, kept
    numerically by accumulating `sum(x)` and `sum(x^T x)` in float64
    regardless of the input dtype (feature values are typically float32;
    accumulating float32 sums-of-squares over tens of thousands of samples
    risks visible precision loss, so this always upcasts before summing).
    """

    def __init__(self, n_features: int) -> None:
        """Start an empty accumulator for `n_features`-dim feature vectors."""
        self.n_features = n_features
        self._n_samples = 0
        self._sum = np.zeros(n_features, dtype=np.float64)
        self._sum_outer = np.zeros((n_features, n_features), dtype=np.float64)

    def add(self, features: np.ndarray) -> None:
        """Fold one trial's (or one block's) raw features into the running sums.

        Args:
            features: `(n_timesteps, n_features)` array, any float dtype.

        Raises:
            ValueError: If `features.shape[-1] != self.n_features`.
        """
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"CovarianceAccumulator.add: expected {self.n_features} "
                f"features, got {features.shape[-1]}."
            )
        x = np.asarray(features, dtype=np.float64)
        self._n_samples += x.shape[0]
        self._sum += x.sum(axis=0)
        self._sum_outer += x.T @ x

    @property
    def n_samples(self) -> int:
        """Total timesteps accumulated so far, across every `add` call."""
        return self._n_samples

    def mean(self) -> np.ndarray:
        """Per-feature mean across every accumulated timestep, `(n_features,)`.

        Raises:
            ValueError: If nothing has been accumulated yet.
        """
        if self._n_samples == 0:
            raise ValueError("CovarianceAccumulator.mean: no samples accumulated.")
        return self._sum / self._n_samples

    def covariance(self) -> np.ndarray:
        """Biased (divide by N, not N-1) covariance, `(n_features, n_features)`.

        Biased rather than the usual N-1-corrected estimator: `N` here is
        typically in the thousands to hundreds of thousands of timesteps
        (`docs/DATASET.md` 20 ms bins), so the N vs. N-1 distinction is
        several orders of magnitude below anything else this module's
        ridge/shrinkage machinery already has to account for -- not worth
        a second code path.

        Raises:
            ValueError: If nothing has been accumulated yet.
        """
        if self._n_samples == 0:
            raise ValueError("CovarianceAccumulator.covariance: no samples accumulated.")
        mean = self.mean()
        return self._sum_outer / self._n_samples - np.outer(mean, mean)


def shrink_covariance(cov: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink a covariance matrix toward a scaled identity.

    `shrunk = (1 - alpha) * cov + alpha * (trace(cov) / p) * I` -- the
    standard shrinkage-toward-identity form (e.g. Ledoit-Wolf's target),
    scaled by `trace(cov) / p` (the average per-feature variance) rather
    than a bare identity, so shrinkage doesn't silently change the overall
    scale of the covariance -- `alpha=1` collapses to an ISOTROPIC matrix
    with the same total variance as `cov`, not a unit-scale one.

    Args:
        cov: `(p, p)` covariance matrix.
        alpha: Shrinkage strength in `[0, 1]`. `0` returns `cov` unchanged;
            `1` returns the fully-isotropic target.

    Returns:
        `(p, p)` shrunk covariance.

    Raises:
        ValueError: If `alpha` is outside `[0, 1]`.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"shrink_covariance: alpha={alpha} must be in [0, 1].")
    p = cov.shape[0]
    target = (np.trace(cov) / p) * np.eye(p)
    return (1.0 - alpha) * cov + alpha * target


@dataclass
class MatrixPowerResult:
    """Diagnostics from `_symmetric_matrix_power`, surfaced to callers so
    Arm 1a/1b can report the condition-number/shrinkage numbers the task
    requires without recomputing an eigendecomposition a second time."""

    sqrt: np.ndarray
    inv_sqrt: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float


def _symmetric_matrix_power(cov: np.ndarray, eps: float = DEFAULT_RIDGE_EPS) -> MatrixPowerResult:
    """Eigendecomposition-based `cov^(1/2)` and `cov^(-1/2)` for a symmetric
    PSD matrix, with a ridge term for numerical stability.

    `np.linalg.eigh` (not the general `eig`) since every covariance here is
    symmetric by construction -- `eigh` is both faster and guarantees real
    eigenvalues, where `eig` could return small spurious complex parts from
    floating-point asymmetry.

    Args:
        cov: `(p, p)` symmetric PSD matrix (a covariance, or a shrunk one).
        eps: Ridge added to every eigenvalue before inverting/rooting
            (`eigenvalues + eps`, never a diagonal added to `cov` itself
            beforehand -- adding directly to the eigenvalues is equivalent
            for a symmetric matrix and avoids re-decomposing after
            perturbing `cov`). Guards `inv_sqrt` against dividing by a
            near-zero or (from floating-point noise) slightly negative
            eigenvalue -- a real risk here, since `cov` can be estimated
            from as few as ~5,000 timesteps for a 512x512 matrix.

    Returns:
        A `MatrixPowerResult` with `sqrt`, `inv_sqrt` (both `(p, p)`,
        symmetric), `eigenvalues` (ascending, the RAW eigenvalues of `cov`
        before the ridge is applied -- what `condition_number` and any
        caller-side "is this ill-conditioned" check should look at,
        since the ridge is a stabilization of the computation, not a
        property of the data worth hiding), and `condition_number`
        (`max(eigenvalues, 0) / max(min(eigenvalues), tiny)` -- the ratio
        that determines how aggressively `inv_sqrt` amplifies the
        smallest-variance directions).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    clipped = np.clip(eigenvalues, a_min=0.0, a_max=None)
    stabilized = clipped + eps
    sqrt_diag = np.sqrt(stabilized)
    inv_sqrt_diag = 1.0 / sqrt_diag

    sqrt_mat = (eigenvectors * sqrt_diag) @ eigenvectors.T
    inv_sqrt_mat = (eigenvectors * inv_sqrt_diag) @ eigenvectors.T

    smallest = max(float(clipped[0]), np.finfo(np.float64).tiny)
    largest = float(clipped[-1])
    condition_number = largest / smallest

    return MatrixPowerResult(
        sqrt=sqrt_mat, inv_sqrt=inv_sqrt_mat, eigenvalues=eigenvalues,
        condition_number=condition_number,
    )


@dataclass
class AlignmentResult:
    """A fitted alignment map plus the diagnostics Phase 3's Task 2 report
    requires (condition number, whether/how much shrinkage was applied)."""

    transform: np.ndarray
    source_condition_number: float
    target_condition_number: float
    target_condition_number_after_shrinkage: float
    shrinkage_alpha: float


def coral_alignment_matrix(
    source_cov: np.ndarray,
    target_cov: np.ndarray,
    eps: float = DEFAULT_RIDGE_EPS,
    shrinkage_alpha: float = 0.0,
) -> AlignmentResult:
    """CORAL: `A = C_target^(-1/2) @ C_source^(1/2)`.

    Args:
        source_cov: `(p, p)` covariance of the training block's features
            (fixed for the whole grid -- computed once from the s05
            training block, never per-session).
        target_cov: `(p, p)` covariance of one session's N calibration
            trials' features (labels discarded).
        eps: Ridge passed to `_symmetric_matrix_power` for both matrices.
        shrinkage_alpha: If > 0, `target_cov` is shrunk toward a scaled
            identity (`shrink_covariance`) BEFORE computing its inverse
            square root -- `source_cov` is never shrunk, since it is
            estimated from the full 908-trial training block and is not
            the small-sample side of this problem.

    Returns:
        An `AlignmentResult` whose `transform` is `A`, `(p, p)`, meant to
        be applied as `X_aligned = X_target @ A` (right-multiplication,
        matching how `DayLayers` and every other per-timestep transform in
        this project applies a `(features,) @ (features, features)` map).
    """
    source_stats = _symmetric_matrix_power(source_cov, eps=eps)

    target_for_fit = target_cov
    if shrinkage_alpha > 0.0:
        target_for_fit = shrink_covariance(target_cov, shrinkage_alpha)
    target_stats = _symmetric_matrix_power(target_for_fit, eps=eps)
    target_stats_unshrunk = _symmetric_matrix_power(target_cov, eps=eps)

    transform = target_stats.inv_sqrt @ source_stats.sqrt
    return AlignmentResult(
        transform=transform,
        source_condition_number=source_stats.condition_number,
        target_condition_number=target_stats_unshrunk.condition_number,
        target_condition_number_after_shrinkage=target_stats.condition_number,
        shrinkage_alpha=shrinkage_alpha,
    )


def procrustes_alignment_matrix(
    source_cov: np.ndarray,
    target_cov: np.ndarray,
    eps: float = DEFAULT_RIDGE_EPS,
    shrinkage_alpha: float = 0.0,
) -> AlignmentResult:
    """Orthogonal Procrustes alignment of two covariances' square roots.

    Solves `min_R ||sqrt_target @ R - sqrt_source||_F` subject to `R`
    orthogonal (Schonemann 1966): `R = U @ Vt` from the SVD of
    `sqrt_target.T @ sqrt_source` (`sqrt_target` is symmetric, so
    `sqrt_target.T == sqrt_target`, but written explicitly for clarity).
    Unlike `coral_alignment_matrix`, `R` is ALWAYS orthogonal
    (`R @ R.T == I`) regardless of `source_cov`/`target_cov` -- it can only
    rotate the target's principal axes onto the source's, never rescale or
    shear (see the module docstring for why that's the point).

    Args:
        source_cov, target_cov, eps, shrinkage_alpha: Same as
            `coral_alignment_matrix`.

    Returns:
        An `AlignmentResult` whose `transform` is `R`, `(p, p)`, applied the
        same way (`X_aligned = X_target @ R`).
    """
    source_stats = _symmetric_matrix_power(source_cov, eps=eps)

    target_for_fit = target_cov
    if shrinkage_alpha > 0.0:
        target_for_fit = shrink_covariance(target_cov, shrinkage_alpha)
    target_stats = _symmetric_matrix_power(target_for_fit, eps=eps)
    target_stats_unshrunk = _symmetric_matrix_power(target_cov, eps=eps)

    cross = target_stats.sqrt.T @ source_stats.sqrt
    u, _, vt = np.linalg.svd(cross)
    transform = u @ vt

    return AlignmentResult(
        transform=transform,
        source_condition_number=source_stats.condition_number,
        target_condition_number=target_stats_unshrunk.condition_number,
        target_condition_number_after_shrinkage=target_stats.condition_number,
        shrinkage_alpha=shrinkage_alpha,
    )


def apply_alignment(features: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a fitted alignment map to a `(n_timesteps, n_features)` block.

    Args:
        features: `(n_timesteps, n_features)` array (or a `(..., n_features)`
            array of any leading shape -- e.g. `(batch, time, n_features)`
            works unchanged, since this is a plain matrix multiply on the
            last axis).
        transform: `(n_features, n_features)`, from `coral_alignment_matrix`
            or `procrustes_alignment_matrix`'s `.transform`.

    Returns:
        `features @ transform`, same shape and dtype as `features`.
    """
    return features @ transform.astype(features.dtype)
