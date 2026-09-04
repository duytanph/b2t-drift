"""Per-session ("day") input adaptation layers.

A `DayLayers` module holds one learned affine transform per recording
session -- `y_t = W[s] @ x_t + b[s]` for every timestep `t` of a trial from
session `s` -- applied to the raw 512-dim feature vector before the shared
recurrent stack. Its purpose is to absorb session-to-session shifts in
neural statistics (electrode drift, impedance changes, day-to-day
recalibration of the recording rig) into a small per-session correction, so
the shared decoder underneath sees a more stationary input distribution
across sessions.

WHY IDENTITY INIT. Every session's `(W, b)` starts at `(I, 0)` -- a no-op.
This matters for two reasons:
  1. A session that has never been trained (no gradient has touched its
     slice yet) must fall back to passing features through unchanged, not
     through a random projection. A random init would actively corrupt
     that session's features until enough gradient steps overwrote it,
     which is never an acceptable default for a slice that might simply
     never get trained (e.g. a held-out evaluation session in this
     project, which never appears in a training batch at all).
  2. It is what makes the Week 3 question -- "how much calibration data
     does a new day need before its day layer is worth using?" -- well
     posed. The answer is naturally bracketed: zero calibration data means
     the identity transform (the session is decoded exactly as if it had
     no day layer at all), and more calibration data moves the layer away
     from identity. If the layer instead started at some arbitrary
     learned or random point, "zero calibration data" would not correspond
     to a meaningful, interpretable baseline.
  This confirms the choice actually made in the published baseline: the
  released pretrained checkpoint's `day_weights.0`..`day_weights.44` are
  identity matrices and `day_biases.0`..`day_biases.44` are zero vectors
  before session-specific fine-tuning (`docs/DESIGN.md` section 7 / Task 4
  checkpoint inspection) -- this module matches that convention, not an
  independent guess.

MEMORY COST. `weights` is `(n_sessions, n_features, n_features)` float32 --
dense, not diagonal or low-rank, since the transform must be able to move
away from identity in any direction during training. At this project's
scale, `n_sessions=45, n_features=512`: `45 * 512 * 512 * 4 bytes ≈
47.2e6 bytes ≈ 47 MB`. That is a large fraction of this model's total
parameter count (the GRU itself, at the default `hidden_size=512,
n_layers=3`, is a few million parameters -- see `GRUCTC.n_parameters()`),
so `DayLayers` is not a negligible add-on; it is sized deliberately, not
by accident, and is worth remembering when reasoning about this model's
total memory footprint or when considering a lower-rank alternative later.

WHY PER-SESSION FREEZING NEEDS A GRADIENT HOOK, NOT `requires_grad`.
Week 3 fits a single new day layer (one session's slice) while holding
everything else -- including every OTHER session's slice of this same
`weights`/`biases` tensor -- frozen. `requires_grad` is a per-TENSOR flag,
not a per-slice one: there is no way to mark "rows 3:4 of this tensor are
trainable, the rest are not" via `requires_grad` alone, since `weights`
and `biases` are each a single `nn.Parameter`. `freeze_all_except` /
`unfreeze_all` therefore work via a gradient hook (`_mask_grad`, registered
once in `__init__` on both `weights` and `biases`) that zeroes every
session's gradient except the active one, applied at backward time, rather
than by touching `requires_grad` at all. Documented explicitly here per
the task that specified this module, since a hook-based "frozen" is a less
obvious mechanism than the usual `requires_grad = False` and is easy to
misread as doing nothing if you don't know to look for it.

CAVEAT: momentum-based optimizers (Adam, the one this project uses) keep a
running average of past gradients per parameter. Freezing a slice zeroes
its FUTURE gradients, but if that slice already had nonzero optimizer
state (`exp_avg`, `exp_avg_sq`) accumulated from steps before
`freeze_all_except` was called, Adam can still apply a few steps of small,
decaying residual updates to the "frozen" slice as that stale state decays
toward zero -- the slice is not necessarily bit-for-bit unchanged the
instant freezing starts. For an exact freeze, call `freeze_all_except`
before any optimizer step has touched the other sessions' slices (e.g.
freeze before the very first `.step()` of a Week 3 fine-tuning run), or
construct a fresh optimizer for the fine-tuning phase so there is no stale
state to decay.
"""

from __future__ import annotations

import torch
from torch import nn


class DayLayers(nn.Module):
    """One learned per-session affine transform, applied before the GRU.

    See the module docstring for why weights/biases initialize to
    identity/zero and why per-session freezing needs a gradient hook.
    """

    def __init__(self, n_sessions: int, n_features: int = 512) -> None:
        """Build one identity-initialized affine transform per session.

        Args:
            n_sessions: Number of sessions to allocate a slice for. Must
                cover every session index this module will ever be called
                with -- `forward` indexes directly into `weights`/`biases`
                with `session_indices` and does not check bounds itself
                (advanced indexing raises `IndexError` on an out-of-range
                index, which is an adequate failure mode here: it names
                the exact offending index via the normal PyTorch/numpy
                error, and re-implementing that check would only
                duplicate it).
            n_features: Feature dimensionality of both the input and
                output of this transform. 512, matching the project's
                fixed raw feature layout (`docs/DATASET.md` section 9) --
                see `GRUCTC`'s docstring for why 512 is not itself a
                tunable hyperparameter in this version.
        """
        super().__init__()
        self.n_sessions = n_sessions
        self.n_features = n_features

        identity = torch.eye(n_features)
        self.weights = nn.Parameter(identity.unsqueeze(0).repeat(n_sessions, 1, 1))
        self.biases = nn.Parameter(torch.zeros(n_sessions, n_features))

        # None = no session is frozen out; every slice trains normally.
        # See the module docstring's "WHY PER-SESSION FREEZING NEEDS A
        # GRADIENT HOOK" section for why this is implemented as a hook
        # registered unconditionally here, rather than toggled on/off in
        # `freeze_all_except`/`unfreeze_all` -- the hook always runs, but
        # is a no-op whenever `_active_session` is None.
        self._active_session: int | None = None
        self.weights.register_hook(self._mask_grad)
        self.biases.register_hook(self._mask_grad)

    def _mask_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Gradient hook: zero every session's slice except the active one.

        Registered on both `weights` and `biases` in `__init__`. Works
        identically for both tensors because masking along dimension 0
        (the session dimension, which is leading in both `(n_sessions,
        n_features, n_features)` and `(n_sessions, n_features)`)
        broadcasts correctly regardless of how many trailing dimensions
        the tensor has.

        Args:
            grad: The gradient PyTorch computed for this parameter,
                same shape as the parameter itself.

        Returns:
            `grad` unchanged if no session is currently frozen
            (`_active_session is None`); otherwise a copy of `grad` with
            every session's slice zeroed except `_active_session`'s.
        """
        if self._active_session is None:
            return grad
        mask = torch.zeros_like(grad)
        mask[self._active_session] = 1.0
        return grad * mask

    def freeze_all_except(self, session_index: int) -> None:
        """Restrict future gradient updates to one session's slice only.

        Every OTHER session's slice of `weights` and `biases` will have
        its gradient zeroed (via `_mask_grad`) on every subsequent
        `.backward()` call, until `unfreeze_all` is called. Does not
        touch `requires_grad` -- see the module docstring for why.

        Args:
            session_index: The one session whose slice should keep
                receiving gradient updates.

        Raises:
            ValueError: If `session_index` is out of `[0, n_sessions)`.
        """
        if not (0 <= session_index < self.n_sessions):
            raise ValueError(
                f"DayLayers.freeze_all_except: session_index={session_index} "
                f"out of range [0, {self.n_sessions})."
            )
        self._active_session = session_index

    def unfreeze_all(self) -> None:
        """Undo `freeze_all_except`: every session's slice trains normally."""
        self._active_session = None

    def forward(self, x: torch.Tensor, session_indices: torch.Tensor) -> torch.Tensor:
        """Apply each batch item's session-specific affine transform.

        Vectorized across the batch via advanced indexing + `torch.bmm` --
        no loop over batch items, so this scales to any batch size at the
        cost of one `(batch, n_features, n_features)` gather (see the
        module docstring's memory-cost note for the per-session tensor
        size this gather draws from).

        Args:
            x: `(batch, time, n_features)` float32.
            session_indices: `(batch,)` int64, each value in
                `[0, n_sessions)`.

        Returns:
            `(batch, time, n_features)` float32, same shape as `x`.
            Identical to `x` for any session whose slice is still at its
            identity initialization (see the module docstring).
        """
        w = self.weights[session_indices]  # (batch, n_features, n_features)
        b = self.biases[session_indices]  # (batch, n_features)
        # y_t = W @ x_t + b, batched: (batch, time, n_features) @ (batch,
        # n_features, n_features) -> (batch, time, n_features). x is
        # row-vector-per-timestep here, so torch.bmm(x, w) computes
        # x_t @ W for every t -- equivalent to (W^T @ x_t^T)^T, i.e. `w`
        # is stored already oriented for this right-multiplication
        # convention (identity init makes the transpose irrelevant at
        # init; it only matters once training moves a slice off identity,
        # and is purely an internal storage convention with no observable
        # effect on what this module computes).
        return torch.bmm(x, w) + b.unsqueeze(1)
