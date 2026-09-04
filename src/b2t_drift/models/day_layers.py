"""Per-session ("day") input adaptation layers.

A `DayLayers` module holds one learned affine transform per recording
session -- `y_t = x_t @ W[s] + b[s]` for every timestep `t` of a trial from
session `s` -- applied to the raw 512-dim feature vector before the shared
recurrent stack. Its purpose is to absorb session-to-session shifts in
neural statistics (electrode drift, impedance changes, day-to-day
recalibration of the recording rig) into a small per-session correction, so
the shared decoder underneath sees a more stationary input distribution
across sessions.

INTERPRETATION (LOAD-BEARING for Week 2 attribution, not just an internal
storage detail): under this convention, `W[s][i, j]` is the weight from
INPUT channel `i` to OUTPUT channel `j`. This must be stated unambiguously
because Week 2 inspects learned day-layer matrices to characterize drift
-- claims like "channel 47's contribution shifted toward channel 52" are
only meaningful if the indexing convention is fixed and known. An earlier
version of this docstring stated the transform as `y_t = W[s] @ x_t +
b[s]` (column-vector convention) while `forward` computed `torch.bmm(x,
w)` (i.e. `x_t @ W`, the TRANSPOSE of that), and called this "a storage
convention with no observable effect." That is true for what the module
COMPUTES (the two conventions differ only by which axis of `W` you call
"input" and which you call "output," and identity init makes them
identical at initialization regardless) but is NOT true for INTERPRETING
a trained `W` -- `W[i, j]` means something different depending on which
convention is in force, and getting this backwards would silently
transpose every attribution claim Week 2 makes. This docstring and
`forward`'s implementation now agree: `y_t = x_t @ W[s] + b[s]`, so
`W[s][i, j]` is unambiguously "how much of output channel j comes from
input channel i."

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
  independent guess. This applies to every `parameterization` below, not
  only `"full"` -- see each variant's identity-init note.

PARAMETERIZATIONS. Day layers are 71% of the model's parameters (11.8M of
16.6M at `n_sessions=45`, default `"full"`), and Week 3 Arm 2 fits one
from as few as 10 labeled sentences -- 262,656 parameters from 10
sentences (roughly 200-500 target timesteps' worth of gradient signal) is
severely underdetermined. `parameterization` trades expressiveness for
parameter count:

  "full" (default) -- `(n_sessions, 512, 512)` `W` + `(n_sessions, 512)`
      bias. 262,656 params/session. Can express ARBITRARY linear
      remapping of the 512 channels, including subspace rotation and
      cross-channel mixing.
  "diagonal" -- per-channel scale only: `(n_sessions, 512)` scale +
      `(n_sessions, 512)` bias, i.e. `y_t = x_t * scale[s] + b[s]`
      (elementwise). 1,024 params/session. Can express independent
      per-channel gain and offset -- e.g. "channel 12 got quieter" --
      but CANNOT express any channel mixing: it has no way to route
      information from one channel into another, by construction.
  "lowrank" -- `W[s] = I + U[s] @ V[s]`, with `U` shape `(n_sessions,
      512, rank)` and `V` shape `(n_sessions, rank, 512)` (a new `rank:
      int = 16` constructor argument), plus bias.
      `512*rank*2 + 512` params/session (16,896 at `rank=16`). Expresses
      a RANK-CONSTRAINED correction to identity: any channel mixing it
      applies is confined to a `rank`-dimensional subspace of the full
      512-dimensional channel space, rather than an arbitrary one.

  ATTRIBUTION CONNECTION (Week 2 / Question 2). Comparing how much PER
  each parameterization recovers, after fitting to the same held-out
  session, is itself a Question 2 result obtained THROUGH the model
  rather than through feature diagnostics: if `"diagonal"` recovers most
  of the lost PER, the drift is well-explained by independent per-channel
  gain/offset changes (consistent with e.g. electrode impedance drift
  acting on each channel separately). If only `"full"` or `"lowrank"`
  meaningfully improves on `"diagonal"`, that is evidence of genuine
  channel MIXING or a subspace ROTATION -- something a per-channel
  explanation cannot produce, since `"diagonal"` is by construction
  incapable of moving information between channels regardless of how
  well it's fit (`tests/test_day_layers.py`'s channel-swap test
  demonstrates this incapability directly, on a synthetic case
  constructed so the correct answer is known).

MEMORY COST. `weights` (the `"full"` parameterization's `W`) is
`(n_sessions, n_features, n_features)` float32 -- dense, not diagonal or
low-rank, since the transform must be able to move away from identity in
any direction during training. At this project's scale, `n_sessions=45,
n_features=512`: `45 * 512 * 512 * 4 bytes ≈ 47.2e6 bytes ≈ 47 MB`. That
is a large fraction of this model's total parameter count (the GRU
itself, at the default `hidden_size=512, n_layers=3`, is a few million
parameters -- see `GRUCTC.n_parameters()`), so `"full"` `DayLayers` is not
a negligible add-on; it is sized deliberately, not by accident. This is
exactly the cost `"diagonal"` and `"lowrank"` exist to reduce -- `47 MB`
becomes `~184 KB` (`"diagonal"`) or `~3 MB` (`"lowrank"`, `rank=16`) at
the same scale.

WHY PER-SESSION FREEZING NEEDS A GRADIENT HOOK, NOT `requires_grad`.
Week 3 fits a single new day layer (one session's slice) while holding
everything else -- including every OTHER session's slice of the SAME
per-session parameter tensor(s) -- frozen. `requires_grad` is a per-TENSOR
flag, not a per-slice one: there is no way to mark "rows 3:4 of this
tensor are trainable, the rest are not" via `requires_grad` alone, since
each per-session tensor (`weights`/`biases` for `"full"`, `scale`/`biases`
for `"diagonal"`, `U`/`V`/`biases` for `"lowrank"`) is a single
`nn.Parameter`. `freeze_all_except` / `unfreeze_all` therefore work via a
gradient hook (`_mask_grad`, registered once in `__init__` on every
per-session parameter this instance owns) that zeroes every session's
gradient except the active one, applied at backward time, rather than by
touching `requires_grad` at all. This works identically for all three
`parameterization` values, since the hook only needs "dimension 0 is the
session dimension," which holds for every per-session tensor here
regardless of its other dimensions. Documented explicitly here since a
hook-based "frozen" is a less obvious mechanism than the usual
`requires_grad = False` and is easy to misread as doing nothing if you
don't know to look for it.

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

`_active_session` IS PART OF `state_dict()`. Stored as a scalar int64
buffer (`-1` meaning "no session frozen," matching `_active_session is
None`) rather than a plain Python attribute, so that saving mid-fine-tune
and reloading preserves which session (if any) is frozen -- a plain
attribute would silently lose that state across a checkpoint round trip,
leaving a reloaded model looking un-frozen even if it was frozen when
saved.
"""

from __future__ import annotations

import math

import torch
from torch import nn

VALID_PARAMETERIZATIONS = ("full", "diagonal", "lowrank")


class DayLayers(nn.Module):
    """One learned per-session transform, applied before the GRU.

    See the module docstring for the transform convention (and why its
    interpretation is load-bearing), the three `parameterization`
    options, why weights/biases initialize to identity/zero, and why
    per-session freezing needs a gradient hook.
    """

    def __init__(
        self,
        n_sessions: int,
        n_features: int = 512,
        parameterization: str = "full",
        rank: int = 16,
    ) -> None:
        """Build one identity-initialized per-session transform.

        Args:
            n_sessions: Number of sessions to allocate a slice for. Must
                cover every session index this module will ever be called
                with -- `forward` indexes directly into the per-session
                parameter tensors with `session_indices` and does not
                check bounds itself (advanced indexing raises
                `IndexError` on an out-of-range index, which is an
                adequate failure mode here: it names the exact offending
                index via the normal PyTorch/numpy error, and
                re-implementing that check would only duplicate it).
            n_features: Feature dimensionality of both the input and
                output of this transform. 512, matching the project's
                fixed raw feature layout (`docs/DATASET.md` section 9) --
                see `GRUCTC`'s docstring for why 512 is not itself a
                tunable hyperparameter in this version.
            parameterization: One of `"full"` (default), `"diagonal"`, or
                `"lowrank"` -- see the module docstring's
                "PARAMETERIZATIONS" section for what each can and cannot
                express, and its "ATTRIBUTION CONNECTION" section for why
                the choice matters beyond parameter count.
            rank: Only used when `parameterization="lowrank"`: the shared
                inner dimension of `U` `(n_sessions, n_features, rank)`
                and `V` `(n_sessions, rank, n_features)`. Default 16 --
                chosen as a round number giving `16,896` params/session
                (a ~15.5x reduction from `"full"`'s `262,656`), not swept
                or otherwise tuned; revisit once Week 3 has real
                small-sample fine-tuning results to compare against.

        Raises:
            ValueError: If `parameterization` is not one of
                `VALID_PARAMETERIZATIONS`, naming both what was given and
                what's accepted; or if `parameterization="lowrank"` and
                `rank < 1`.
        """
        super().__init__()
        if parameterization not in VALID_PARAMETERIZATIONS:
            raise ValueError(
                f"DayLayers.__init__: parameterization={parameterization!r} "
                f"not in {VALID_PARAMETERIZATIONS}."
            )
        if parameterization == "lowrank" and rank < 1:
            raise ValueError(
                f"DayLayers.__init__: rank={rank} must be >= 1 when "
                f"parameterization='lowrank'."
            )

        self.n_sessions = n_sessions
        self.n_features = n_features
        self.parameterization = parameterization
        self.rank = rank

        # -1 = no session is frozen out; every slice trains normally.
        # A registered buffer (not a plain attribute) so freeze state
        # round-trips through state_dict() -- see the module docstring's
        # "_active_session IS PART OF state_dict()" section. Exposed as
        # the `_active_session` property below (int-or-None) so the rest
        # of this class's logic reads the same as before this was made
        # persistent.
        self.register_buffer("_active_session_buffer", torch.tensor(-1, dtype=torch.int64))

        if parameterization == "full":
            identity = torch.eye(n_features)
            self.weights = nn.Parameter(identity.unsqueeze(0).repeat(n_sessions, 1, 1))
            self.biases = nn.Parameter(torch.zeros(n_sessions, n_features))
            self._session_params: tuple[nn.Parameter, ...] = (self.weights, self.biases)
        elif parameterization == "diagonal":
            # Identity: scale=1, bias=0 -- see the module docstring's
            # "WHY IDENTITY INIT."
            self.scale = nn.Parameter(torch.ones(n_sessions, n_features))
            self.biases = nn.Parameter(torch.zeros(n_sessions, n_features))
            self._session_params = (self.scale, self.biases)
        else:  # "lowrank"
            # I + U @ V = I when U = 0, regardless of V -- so U is the
            # factor that must start at zero. V is NOT also zeroed:
            # d(loss)/dV = U^T @ (upstream gradient), so if U were also
            # zero, V would receive exactly zero gradient too (a
            # classic dead-symmetry saddle point -- both factors would
            # stay at zero forever, since each one's gradient depends on
            # the other being nonzero). V instead gets a small random
            # init (std 1/sqrt(n_features), a standard small-init scale)
            # so U (which DOES get nonzero gradient from a nonzero V)
            # can start learning immediately; the product U @ V is still
            # exactly zero at t=0 either way, since U=0. This mirrors
            # the standard LoRA-style low-rank init convention (one
            # factor zero, the other small-random) for exactly this
            # symmetry-breaking reason.
            self.U = nn.Parameter(torch.zeros(n_sessions, n_features, rank))
            self.V = nn.Parameter(
                torch.randn(n_sessions, rank, n_features) / math.sqrt(n_features)
            )
            self.biases = nn.Parameter(torch.zeros(n_sessions, n_features))
            self._session_params = (self.U, self.V, self.biases)

        for param in self._session_params:
            param.register_hook(self._mask_grad)

    @property
    def _active_session(self) -> int | None:
        """The one session currently exempt from gradient masking, or None."""
        value = int(self._active_session_buffer.item())
        return None if value < 0 else value

    @_active_session.setter
    def _active_session(self, value: int | None) -> None:
        self._active_session_buffer.fill_(-1 if value is None else value)

    def _mask_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Gradient hook: zero every session's slice except the active one.

        Registered on every per-session parameter tensor this instance
        owns (see `__init__`). Works identically regardless of
        `parameterization` or how many trailing dimensions the tensor
        has, because masking always happens along dimension 0 (the
        session dimension, which is leading in every per-session tensor
        here), which broadcasts correctly no matter what.

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

        Every OTHER session's slice of every per-session parameter this
        instance owns will have its gradient zeroed (via `_mask_grad`) on
        every subsequent `.backward()` call, until `unfreeze_all` is
        called. Does not touch `requires_grad` -- see the module
        docstring for why.

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
        """Apply each batch item's session-specific transform.

        Vectorized across the batch via advanced indexing + batched
        matmul (`torch.bmm`) -- no loop over batch items, so this scales
        to any batch size at the cost of one per-session parameter
        gather (see the module docstring's memory-cost note for how that
        gather's size depends on `parameterization`).

        Args:
            x: `(batch, time, n_features)` float32.
            session_indices: `(batch,)` int64, each value in
                `[0, n_sessions)`.

        Returns:
            `(batch, time, n_features)` float32, same shape as `x`.
            Identical to `x` for any session whose slice is still at its
            identity initialization (see the module docstring), for
            every `parameterization`.
        """
        if self.parameterization == "full":
            w = self.weights[session_indices]  # (batch, n_features, n_features)
            b = self.biases[session_indices]  # (batch, n_features)
            # y_t = x_t @ W[s] + b[s] -- see the module docstring's
            # INTERPRETATION note: W[s][i, j] is the weight from input
            # channel i to output channel j under this convention.
            return torch.bmm(x, w) + b.unsqueeze(1)

        if self.parameterization == "diagonal":
            scale = self.scale[session_indices].unsqueeze(1)  # (batch, 1, n_features)
            b = self.biases[session_indices].unsqueeze(1)  # (batch, 1, n_features)
            # Elementwise -- no channel mixing is possible by
            # construction (see the module docstring's "PARAMETERIZATIONS").
            return x * scale + b

        # "lowrank": y_t = x_t @ (I + U[s] @ V[s]) + b[s]
        #          = x_t + (x_t @ U[s]) @ V[s] + b[s]
        # Computed via two small batched matmuls (never materializing
        # the full n_features x n_features matrix) -- this is the whole
        # point of the low-rank parameterization.
        u = self.U[session_indices]  # (batch, n_features, rank)
        v = self.V[session_indices]  # (batch, rank, n_features)
        b = self.biases[session_indices].unsqueeze(1)  # (batch, 1, n_features)
        return x + torch.bmm(torch.bmm(x, u), v) + b
