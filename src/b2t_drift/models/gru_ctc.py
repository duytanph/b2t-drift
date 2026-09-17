"""GRU + CTC phoneme decoder.

The single model family used in this project: a per-session `DayLayers`
affine adaptation, a fixed Gaussian temporal smoothing filter, a `nn.GRU`
encoder, and a linear classifier trained with CTC loss against phoneme
label sequences. No other architectures are planned -- see `CLAUDE.md` on
avoiding premature abstraction.

ARCHITECTURE, in forward order:
  1. (optional) `input_dropout`: channel dropout on the raw 512-dim input,
     simulating electrode failure. Off by default -- see `__init__`.
  2. (optional) `DayLayers`: per-session affine transform. See
     `models/day_layers.py`.
  3. PADDING MASK: every timestep at or past each item's `input_lengths[i]`
     is zeroed. MANDATORY, not optional -- see "PADDING MUST BE MASKED
     BEFORE SMOOTHING" below for why this step exists and must run here,
     between `DayLayers` and smoothing, not anywhere else.
  4. Gaussian temporal smoothing: fixed, non-learned depthwise `conv1d`
     over the time axis. See `_smooth`.
  5. `nn.GRU`: the shared recurrent encoder. Uses PyTorch's built-in,
     cuDNN-backed implementation directly -- NOT a hand-rolled per-timestep
     loop, which would be 10-50x slower on GPU because it can't use
     cuDNN's fused RNN kernels. Do not "unroll" this for debuggability;
     use `torch.autograd` / a debugger on the fused op instead. Always run
     through `pack_padded_sequence`/`pad_packed_sequence` (`forward`'s
     `input_lengths` is mandatory) so padded timesteps never enter the
     recurrence.
  6. A linear projection to `n_classes` logits, then `log_softmax` over the
     class dimension.

PADDING MUST BE MASKED BEFORE SMOOTHING (fixes a verified bug -- numbers
below). `DayLayers` runs on EVERY timestep, including collate-padded ones,
and once its bias has moved off its zero initialization (i.e. after any
training), applying it to a padded (zero) timestep produces `0 @ W + bias
= bias` -- a NONZERO constant, not the zero the padding started as.
`_smooth`'s depthwise conv then mixes that leaked constant into the last
few real timesteps near each sequence's boundary. Because how much padding
a given trial gets depends on which OTHER trials share its batch,
THE SAME TRIAL could decode differently depending on batch composition --
silently, since nothing here raised. Measured before this fix: 0.0755 max
absolute log-probs difference for a trained (nonzero-bias) `DayLayers`,
comparing one trial decoded alone versus padded inside a larger batch --
large enough to flip a marginal argmax, i.e. large enough to actually
change the decoded phoneme sequence, not just perturb a log-probability by
a harmless amount. THE FIX: zero every padded timestep immediately after
`DayLayers`, before `_smooth` ever sees it -- collate's zero-padding and
`_smooth`'s own implicit zero edge-padding are then genuinely identical
inputs to the convolution again (as they always were for an untrained
`DayLayers`; see `_smooth`'s docstring), regardless of what `DayLayers`
has learned. Measured after this fix: < 1e-5 (see
`tests/test_gru_ctc.py::test_smoothing_batch_composition_sensitivity_reports_magnitude`
for the exact reproduced value). Recorded with both numbers in
`docs/DESIGN.md`'s "Model hyperparameters" section.

THE CTC LENGTH CONTRACT (see also `data/collate.py`'s module docstring,
which this model's caller must also have read). `patch_size`/`patch_stride`
(default `1`/`1`, a no-op) control whether the GRU consumes one output per
INPUT timestep (the original, no-downsampling behavior) or one output per
PATCH of `patch_size` consecutive timesteps stepped every `patch_stride`
-- `patch_size=14, patch_stride=4` matches the published baseline (a
7,168-dim input, `14*512`, at an effective 80 ms output stride). Whenever
`patch_size > 1` or `patch_stride > 1`, `input_lengths` MUST be divided
down to the patched time axis before `nn.CTCLoss` ever sees them
(`CLAUDE.md`'s "known pitfalls" -- getting this wrong is the single most
likely place for a silent bug in this whole pipeline, since a wrong length
doesn't raise, it just corrupts the loss: the model trains, converges, and
emits garbage). `ctc_input_lengths` is the ONE place this conversion
happens -- `forward` itself calls it (for `pack_padded_sequence` after
patching, see `_patch`'s note on `patched_lengths`), and every external
caller (`training/loop.py`) routes `input_lengths` through it before
`nn.CTCLoss`, so there is exactly one formula to get right, used
identically on both sides.

NOTE: `ctc_input_lengths` (the CTC-loss length adjustment) and `forward`'s
optional `input_lengths` argument (padding masking around the GRU) are
TWO DIFFERENT USES of the same underlying per-trial timestep counts from
`collate_trials`. They happen to be the identity function's input and a
`pack_padded_sequence` call's input respectively, in this no-downsampling
version, but they answer different questions ("how long is this
sequence for the loss" vs. "where does padding start for the RNN") and
must both be threaded through independently -- fixing one does not fix
the other.

THE TIME-FIRST OUTPUT CONVENTION. `forward` returns `log_probs` shaped
`(time, batch, n_classes)`, matching `torch.nn.CTCLoss`'s required
`log_probs` layout directly -- CALLERS MUST NOT TRANSPOSE, AND MUST NOT
APPLY `log_softmax` AGAIN. `log_softmax` is applied exactly once, inside
this `forward`, over the class dimension; applying it a second time
outside would silently produce a wrong (still positive-summing-to-more-
than-1-in-log-space... more precisely, doubly-normalized and numerically
wrong) loss without `nn.CTCLoss` raising anything, since it takes
log-probabilities on faith and never itself checks that they came from
exactly one `log_softmax`.

WHY `bidirectional=True` IS STILL REFUSED, EVEN THOUGH `input_lengths` IS
NOW MANDATORY. The padding mask above plus the now-mandatory
`pack_padded_sequence`/`pad_packed_sequence` around the GRU together close
the CORRECTNESS gap that originally motivated refusing
`bidirectional=True` in `__init__` (a bidirectional GRU's
backward-direction pass used to be able to run through un-masked padding
into every real timestep; packed sequences prevent that regardless of
direction, since cuDNN's packed kernels already know each item's true
length). The refusal remains in place anyway, because it was always ALSO
a `[principled]` scope decision independent of that correctness gap: this
project measures a realistically-configured, causal decoder (see
`bidirectional`'s `__init__` docstring entry) -- re-enabling bidirectional
support is a smaller lift now than before, but is a deliberate future
change, not implied by this fix.

DEVIATIONS FROM THE PUBLISHED BASELINE, gathered in one place:
  - Plain 512-dim input at 20 ms bins by DEFAULT (`patch_size=1,
    patch_stride=1`); 14-bin/stride-4 patching (see the CTC length
    contract note above) is now implemented and available via
    `patch_size=14, patch_stride=4`, but is opt-in, not the default --
    an explicit later comparison (`docs/DESIGN.md` section 7's Task 3),
    not inherited automatically.
  - `n_layers=3` vs. the baseline's 5 (smaller training set here).
  - `hidden_size=512` vs. the baseline's 768 (same reason).
  - Unidirectional (causal) by default; the baseline (an offline research
    checkpoint, not the realtime deployed system) does not face this
    constraint the same way -- see `bidirectional`'s docstring entry
    below for the full argument.
  All four are recorded again, with tags, in each hyperparameter's
  docstring entry below and in `docs/DESIGN.md`'s "Model hyperparameters"
  section -- this list is a pointer to that reasoning, not a duplicate of
  it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from b2t_drift.models.day_layers import DayLayers

DEFAULT_N_FEATURES = 512
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_N_LAYERS = 3
DEFAULT_N_CLASSES = 41
DEFAULT_DROPOUT = 0.4
DEFAULT_SMOOTHING_SIGMA_BINS = 2.0


def _gaussian_kernel1d(sigma: float) -> torch.Tensor:
    """Build a 1-D, unit-sum Gaussian smoothing kernel.

    Args:
        sigma: Standard deviation, in bins. Must be > 0 (callers gate on
            `smoothing_sigma_bins > 0` before calling this -- there is no
            sensible kernel for sigma=0, which is handled as "smoothing
            disabled" instead, never as a degenerate 1-tap kernel here).

    Returns:
        `FloatTensor`, shape `(kernel_size,)`, `kernel_size = 2*radius+1`
        with `radius = ceil(3*sigma)` (covers +/-3 standard deviations,
        the conventional truncation point beyond which a Gaussian's mass
        is negligible), summing to 1.0 so smoothing does not change the
        overall scale of the signal.
    """
    radius = max(1, math.ceil(3 * sigma))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


class GRUCTC(nn.Module):
    """GRU encoder with a linear CTC output head, plus day-layer adaptation
    and fixed Gaussian temporal smoothing.

    See the module docstring for the full architecture, the CTC length
    contract, the time-first output convention, and why padding masking
    matters more for `bidirectional=True` than the unidirectional default.
    """

    def __init__(
        self,
        n_sessions: int,
        n_features: int = DEFAULT_N_FEATURES,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        n_layers: int = DEFAULT_N_LAYERS,
        n_classes: int = DEFAULT_N_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        bidirectional: bool = False,
        use_day_layers: bool = True,
        input_dropout: float = 0.0,
        smoothing_sigma_bins: float = DEFAULT_SMOOTHING_SIGMA_BINS,
        patch_size: int = 1,
        patch_stride: int = 1,
        day_layer_parameterization: str = "full",
        day_layer_rank: int = 16,
    ) -> None:
        """Build the GRU/CTC decoder.

        Every default below is a deliberate project decision, not an
        arbitrary PyTorch default -- each entry gives (i) the value,
        (ii) the published baseline's value where one exists, (iii) the
        reason, and (iv) a tag: `[principled]` (derived from a physical
        or architectural argument), `[convention]` (standard in this
        specific literature), `[judgment call]` (a reasonable choice
        given constraints, not derived from first principles -- revisit
        if named), or `[fixed by data]` (not a free hyperparameter at
        all). Also recorded, with the same tags, in `docs/DESIGN.md`'s
        "Model hyperparameters" section, so the reasoning lives outside
        the source too.

        Args:
            n_sessions: Number of sessions `DayLayers` should allocate a
                slice for (ignored if `use_day_layers=False`). No default
                -- this is data-dependent (45 sessions in the full T15
                dataset, 2 in the pipeline gate test's synthetic setup;
                `docs/DATASET.md`), never a fixed constant.
            n_features: 512. Input neural feature dimensionality: 256
                threshold-crossing + 256 spike-band-power channels at
                20 ms bins, the released HDF5 `input_features` layout
                with no patching or stride applied (see the module
                docstring's "deviations from the published baseline").
                Not a free choice -- it's the shape the data comes in.
                `[fixed by data]`
            hidden_size: 512. Pretrained checkpoint uses 768. Reduced for
                the same data-budget reason as `n_layers` below. 512 also
                matches the input dimension, which keeps the first layer
                square. `[judgment call — sweep this]`
            n_layers: 3. Baseline uses 5. Reduced because our
                Switchboard-only training set is 3,150 trials versus
                roughly 8,000 for the baseline; depth is the first thing
                to cut when data is limited. Three is also the common
                floor for sequence models of this kind — below that,
                RNNs tend to underfit temporal structure.
                `[judgment call — revisit if the baseline underfits]`
            n_classes: 41. Not a hyperparameter. CTC blank (0) + 39
                ARPABET phonemes (1-39) + silence (40), verified
                empirically against `seq_class_ids`
                (`b2t_drift.phonemes`). `[fixed by data]`
            dropout: 0.4. High by general ML standards, standard in this
                specific literature. Willett's speech RNN and its
                descendants use heavy regularization because the
                datasets are small and the input is noisy. Passed
                straight through to `nn.GRU`'s own `dropout` argument,
                so (per `nn.GRU`'s semantics) it has no effect when
                `n_layers=1` -- there is no dropout to apply between a
                single layer and itself. `[convention]`
            bidirectional: False. A bidirectional GRU sees the entire
                utterance before emitting, which is fine offline but
                impossible in a real-time system. The deployed system
                emits a phoneme every 80 ms while the participant is
                still speaking. Since this project measures a
                realistically-configured decoder, causal is the honest
                default. Bidirectional would improve offline PER and is
                a legitimate ablation, but must not be the default.
                **`True` is currently REFUSED (raises `ValueError` right
                here in `__init__`)**. Originally refused because a
                bidirectional GRU's backward-direction pass could run
                through un-masked padding into every real timestep; now
                that `forward`'s `input_lengths` is MANDATORY (padding is
                always masked and the GRU is always packed -- see the
                module docstring's "PADDING MUST BE MASKED BEFORE
                SMOOTHING"), that correctness gap is closed. The refusal
                stays anyway as a deliberate scope decision: this project
                measures a realistically-configured, causal decoder, so
                bidirectional remains a future ablation to enable
                explicitly, not a default to fall into.
                `[principled]`
            use_day_layers: True. Whether to apply `DayLayers` at all.
                Not one of the four explicitly-justified hyperparameters
                above, but documented here for the same reason: the
                published architecture includes per-session adaptation,
                and this project's Week 3 question ("how much
                calibration data does a new day need") is specifically
                about this component, so it defaults on. Set False for
                an ablation isolating the GRU/CTC core's own behavior,
                or (as in the pipeline gate test) to construct a model
                with zero day-layer parameters. `[principled]`
            input_dropout: 0.0. Off by default. Channel dropout
                simulating electrode failure is a Week 2 experiment, not
                a training default — enabling it now would confound the
                baseline. When enabled, drops entire feature channels
                (via `nn.Dropout1d`, not elementwise `nn.Dropout`) for
                the whole trial at once, matching what a truly failed
                electrode would look like -- intermittent per-timestep
                dropout would not simulate that failure mode. Applied
                only in training mode (`nn.Dropout1d`'s own behavior),
                before the day layer. `[judgment call]`
            smoothing_sigma_bins: 2.0. From the published pipeline
                (sigma = 2 bins = 40 ms). Physical argument: articulatory
                gestures last roughly 50-150 ms, so 40 ms of smoothing
                averages away per-bin spike-count variance without
                blurring the gestures being decoded. Both documented in
                the literature and derivable from the timescale of
                speech. Set to 0 to disable smoothing entirely (identity
                -- see `_smooth`). `[principled]`
            patch_size: 1 (no-op -- plain per-timestep input, matching
                every default above). 14 matches the published baseline:
                concatenate 14 consecutive 20 ms bins into one GRU input
                vector (`14 * 512 = 7,168`-dim). See `_patch` for the
                exact concatenation order and `ctc_input_lengths` for the
                matching length-conversion formula -- the two MUST use
                the same `patch_size`/`patch_stride` (both read from
                `self`, so they always do) or the CTC loss silently sees
                the wrong length (`CLAUDE.md`'s "known pitfalls"; the
                module docstring's "THE CTC LENGTH CONTRACT"). An
                explicit later comparison (`docs/DESIGN.md` section 7's
                Task 3), not inherited as a default. `[principled]` when
                set to 14 (matches the published pipeline's real-time
                80 ms output-stride design); the default of 1 is a
                `[judgment call]` scope decision (start with no patching,
                add it as a controlled comparison), not derived.
            patch_stride: 1 (no-op, paired with `patch_size=1`). 4
                matches the published baseline (stepped every 4 of the
                20 ms bins that `patch_size` groups together, i.e. every
                80 ms). See `patch_size`'s entry -- same tags, same
                reasoning.
            day_layer_parameterization: `"full"` (default), `"diagonal"`,
                or `"lowrank"` -- forwarded to `DayLayers`; ignored when
                `use_day_layers=False`. See `DayLayers`' module docstring
                for what each can express and its parameter cost.
                `[judgment call]` -- `"full"` matches the published
                checkpoint's `day_weights` (512x512 per session), but
                `docs/FINDINGS.md` ("Day layers are unstable when trained
                jointly across many sessions") records that `"full"`
                destabilizes joint training at this project's scale;
                `"diagonal"` is the leaner alternative Arm 2 would
                actually ship.
            day_layer_rank: Only used when
                `day_layer_parameterization="lowrank"`; forwarded to
                `DayLayers`' `rank`. Default 16.

        Raises:
            ValueError: If `bidirectional=True` (not yet supported --
                see `bidirectional`'s entry above); or if `patch_size < 1`
                or `patch_stride < 1` (both must be positive -- `1` is
                the no-op floor, not `0`, since a patch of size 0 or a
                stride of 0 has no sensible meaning).
        """
        super().__init__()

        if bidirectional:
            raise ValueError(
                "GRUCTC.__init__: bidirectional=True is not yet "
                "supported. forward()'s padding mask and mandatory "
                "pack_padded_sequence/pad_packed_sequence now prevent a "
                "bidirectional GRU's backward-direction pass from "
                "reading padding regardless of direction, so the "
                "original correctness concern is closed -- this remains "
                "refused as a deliberate scope decision (this project "
                "measures a realistically-configured, causal decoder; "
                "see bidirectional's docstring entry), not a technical "
                "limitation, and is a smaller lift to re-enable than it "
                "used to be."
            )
        if patch_size < 1 or patch_stride < 1:
            raise ValueError(
                f"GRUCTC.__init__: patch_size={patch_size} and patch_stride={patch_stride} "
                f"must both be >= 1 (1 is the no-op floor -- a patch of size 0 or a stride "
                f"of 0 has no sensible meaning)."
            )

        self.n_sessions = n_sessions
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.use_day_layers = use_day_layers
        self.input_dropout = input_dropout
        self.smoothing_sigma_bins = smoothing_sigma_bins
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.day_layer_parameterization = day_layer_parameterization
        self.day_layer_rank = day_layer_rank

        self._input_dropout = nn.Dropout1d(input_dropout) if input_dropout > 0 else None

        self.day_layers = (
            DayLayers(
                n_sessions,
                n_features,
                parameterization=day_layer_parameterization,
                rank=day_layer_rank,
            )
            if use_day_layers
            else None
        )

        if smoothing_sigma_bins > 0:
            kernel = _gaussian_kernel1d(smoothing_sigma_bins)
            self._smoothing_radius = (kernel.numel() - 1) // 2
            # Depthwise conv1d weight layout: (out_channels, in_channels /
            # groups, kernel_size) = (n_features, 1, kernel_size) with
            # groups=n_features -- one identical 1-D kernel applied
            # independently per feature channel. A buffer, not a
            # Parameter: this filter is fixed for the life of the model
            # (never appears in .parameters(), never receives a gradient,
            # never shows up in n_parameters()'s counts) and must still
            # move with the module under .to(device)/.cuda(), which is
            # exactly what register_buffer is for.
            self.register_buffer(
                "_smoothing_kernel",
                kernel.view(1, 1, -1).repeat(n_features, 1, 1),
            )
        else:
            self._smoothing_radius = 0
            # Registered as a (None-valued) buffer, NOT a plain instance
            # attribute, even though there is no tensor to store yet.
            # Fixes a real inconsistency: a plain `self._smoothing_kernel
            # = None` here is invisible to `nn.Module` machinery (never
            # appears in `self._buffers`, `named_buffers()`, or
            # `._apply()`/`.to(device)`'s traversal), so IF this attribute
            # were ever reassigned a real tensor after construction, it
            # would silently stay an untracked plain attribute forever
            # rather than a proper buffer. `register_buffer(..., None)`
            # closes that gap: `_smoothing_kernel` is now a registered
            # buffer slot either way, so a later `self._smoothing_kernel
            # = some_tensor` assignment updates the SAME registered slot,
            # consistent with the smoothing-enabled branch.
            #
            # CAVEAT, stated precisely rather than overclaimed: this does
            # NOT make `state_dict()`'s key SET identical between
            # smoothing-on and smoothing-off models -- PyTorch's
            # `state_dict()`/`named_buffers()` both skip a buffer whose
            # current value is None (verified empirically; a None buffer
            # never round-trips through `state_dict()`/`load_state_dict`
            # regardless of registration), so `state_dict()` still omits
            # `_smoothing_kernel` entirely when smoothing is off. What
            # this fix actually guarantees is narrower and still real:
            # `_smoothing_kernel` is a known buffer NAME in
            # `self._buffers` in both configurations (see
            # `tests/test_gru_ctc.py`'s buffer-consistency test), so
            # every OTHER state_dict key (day_layers.*, gru.*, output.*)
            # is unaffected by this toggle, and the module's own
            # attribute-tracking machinery treats the two configurations
            # uniformly rather than one being a real buffer and the other
            # an ad hoc Python attribute.
            self.register_buffer("_smoothing_kernel", None)

        self.gru = nn.GRU(
            # patch_size=1 (the default) makes this exactly n_features,
            # matching the pre-patching model's GRU exactly -- see
            # `_patch`'s docstring for the concatenation order this
            # dimension assumes.
            input_size=n_features * patch_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.output = nn.Linear(gru_output_size, n_classes)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fixed Gaussian temporal smoothing, or return `x` unchanged.

        Applied in BOTH training and eval modes -- unlike dropout, this
        is not a regularizer that should only act during training; it
        matches the published pipeline's fixed preprocessing step, which
        is a property of the FEATURES, not of the training regime.

        BATCH-COMPOSITION-DEPENDENCE: FIXED IN `forward`, NOT HERE.
        Callers reaching this method should already have zeroed every
        padded timestep -- `forward` does so immediately after
        `DayLayers` and before calling `_smooth` (see the module
        docstring's "PADDING MUST BE MASKED BEFORE SMOOTHING"). With
        that precondition, this method's own conv1d boundary behavior is
        provably batch-invariant: `F.conv1d`'s `padding=radius` argument
        zero-fills the edge of whatever tensor it's given, so a trial
        processed ALONE (tensor length == that trial's own length) and
        the SAME trial's real-data region processed PADDED inside a
        larger batch present IDENTICAL values (zero, either way) to the
        convolution at the boundary, regardless of what `DayLayers` has
        learned -- confirmed empirically, < 1e-5 max abs difference for a
        TRAINED (nonzero-bias) `DayLayers`, down from 0.0755 before
        `forward` masked the padding (see
        `tests/test_gru_ctc.py::test_smoothing_batch_composition_sensitivity_reports_magnitude`
        and `docs/DESIGN.md`'s "Model hyperparameters" section for both
        numbers). PRIOR TO THAT FIX, this method silently mixed a
        trained `DayLayers`' leaked bias into the last few real
        timesteps whenever a trial was batched with a longer one -- this
        docstring used to document that as an accepted limitation of
        `_smooth` itself; it is now understood (and fixed) as a bug in
        `forward`'s masking order, not an inherent property of
        smoothing. `_smooth` NEVER masks anything itself -- it trusts
        its caller to have already zeroed padding, which is why calling
        this method directly (bypassing `forward`) on unmasked input
        would reintroduce the old bug.

        Args:
            x: `(batch, time, n_features)` float32.

        Returns:
            `(batch, time, n_features)` float32, same shape as `x`.
            Identical to `x` (up to floating-point roundoff from the
            conv/matmul path) if `smoothing_sigma_bins == 0`.
        """
        if self._smoothing_kernel is None:
            return x
        # conv1d expects (batch, channels, time); x is (batch, time,
        # channels) throughout this model (see the module docstring),
        # so transpose in, convolve, transpose back.
        x_bft = x.transpose(1, 2)  # (batch, n_features, time)
        smoothed = F.conv1d(
            x_bft,
            self._smoothing_kernel,
            padding=self._smoothing_radius,
            groups=self.n_features,
        )
        return smoothed.transpose(1, 2)  # (batch, time, n_features)

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate `patch_size` consecutive timesteps, stepped every
        `patch_stride`, or return `x` unchanged if `patch_size==1 and
        patch_stride==1` (the default -- no-op).

        MUST run AFTER `_smooth`, never before: `smoothing_sigma_bins`
        means "2 bins = 40 ms" at the ORIGINAL 20 ms-per-bin resolution
        (the module docstring's hyperparameter table). Smoothing across
        PATCHED (80 ms-stride) positions instead would smooth over a
        completely different, much coarser timescale -- see `forward`'s
        call order for where this is enforced.

        CONCATENATION ORDER. `x.unfold(1, patch_size, patch_stride)`
        gives `(batch, n_patches, n_features, patch_size)` -- PyTorch's
        `unfold` always appends the windowed dimension LAST. Flattening
        that directly would interleave as "all `patch_size` values of
        feature 0, then all `patch_size` values of feature 1, ..."
        (feature-major), which is NOT what "concatenate 14 consecutive
        20 ms bins" (the published baseline's own description) means --
        that means BIN-major: bin 0's full 512-dim feature vector, then
        bin 1's, ..., then bin 13's, concatenated into one 7,168-dim
        vector. This method `.permute`s to `(batch, n_patches,
        patch_size, n_features)` BEFORE flattening, specifically to get
        that bin-major order.

        Args:
            x: `(batch, time, n_features)` float32, already smoothed and
                padding-masked.

        Returns:
            `x` unchanged if `patch_size==1 and patch_stride==1`.
            Otherwise `(batch, n_patches, patch_size * n_features)`
            float32, where `n_patches = (time - patch_size) //
            patch_stride + 1` -- EXACTLY the formula `ctc_input_lengths`
            applies to lengths (both must agree, and PyTorch's `unfold`
            is documented to produce exactly this many windows, so the
            two are guaranteed consistent rather than independently
            re-derived).
        """
        if self.patch_size == 1 and self.patch_stride == 1:
            return x
        patches = x.unfold(1, self.patch_size, self.patch_stride)  # (batch, n_patches, n_features, patch_size)
        patches = patches.permute(0, 1, 3, 2).contiguous()  # (batch, n_patches, patch_size, n_features)
        batch, n_patches = patches.shape[0], patches.shape[1]
        return patches.reshape(batch, n_patches, self.patch_size * self.n_features)

    def ctc_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map raw per-trial `input_lengths` to CTC-ready (post-patching)
        lengths.

        The IDENTITY function when `patch_size==1 and patch_stride==1`
        (the default): this configuration applies no temporal
        downsampling anywhere (see the module docstring's CTC length
        contract), so `nn.CTCLoss`'s `input_lengths` argument is
        exactly the same as `collate_trials`'s `input_lengths`.

        WHEN `patch_size > 1` OR `patch_stride > 1`: applies the SAME
        formula `_patch`'s `unfold` call uses internally to determine
        `n_patches` (PyTorch's own documented `unfold` behavior), via
        integer floor division (`rounding_mode="floor"`, never float
        division -- avoids any floating-point rounding disagreement with
        `unfold`'s own integer arithmetic, which matters exactly at
        multiples where float division could round the wrong way):

            ctc_len = (input_len - patch_size) // patch_stride + 1

        Callers should ALWAYS route `input_lengths` through this method
        before passing them to `nn.CTCLoss` -- this discipline is what
        prevents the length adjustment from being silently forgotten at
        whichever call site changes `patch_size`/`patch_stride` later
        (`CLAUDE.md`'s "known pitfalls"), and is also what `forward`
        itself does internally (see its `patched_lengths` note) so the
        GRU's `pack_padded_sequence` and `nn.CTCLoss` always agree on
        exactly the same per-item length.

        NOTE: unrelated to `forward`'s own `input_lengths` argument (which
        controls `pack_padded_sequence`/`pad_packed_sequence` around the
        GRU) other than sharing the same underlying tensor as their input
        -- see the module docstring's note on this.

        Args:
            input_lengths: `(batch,)` int64, from `collate_trials`.

        Returns:
            `(batch,)` int64. `input_lengths` unchanged (the same tensor
            object, not a copy) when `patch_size==1 and patch_stride==1`;
            otherwise a new tensor per the formula above.
        """
        if self.patch_size == 1 and self.patch_stride == 1:
            return input_lengths
        return torch.div(input_lengths - self.patch_size, self.patch_stride, rounding_mode="floor") + 1

    def _run_gru(self, packed: nn.utils.rnn.PackedSequence) -> nn.utils.rnn.PackedSequence:
        """Run `self.gru` on an already-packed input, initial hidden state
        zero (PyTorch's `nn.GRU` default).

        Extracted as its own method ONLY so `PretrainedBaselineRNN`
        (`models/pretrained_baseline.py`) can override it to inject the
        official checkpoint's learned initial hidden state (`h0`, a
        parameter this project's own architecture has no equivalent of --
        see that module's docstring) without duplicating `forward`'s
        surrounding padding-mask/smooth/patch logic. Not meant to be
        overridden for any other reason.

        Args:
            packed: Output of `nn.utils.rnn.pack_padded_sequence`.

        Returns:
            The GRU's packed output (hidden state discarded -- nothing in
            this codebase's forward pass uses the final hidden state).
        """
        packed_out, _ = self.gru(packed)
        return packed_out

    def forward(
        self,
        features: torch.Tensor,
        session_indices: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-timestep log-class-probabilities.

        Args:
            features: `(batch, time, 512)` float32, from
                `collate_trials`'s `"features"`.
            session_indices: `(batch,)` int64, from `collate_trials`'s
                `"session_indices"`. Read even when `use_day_layers=False`
                (accepted for a uniform call signature) but unused in
                that case.
            input_lengths: `(batch,)` int64. Each item's true (unpadded)
                timestep count -- the SAME `collate_trials`
                `"input_lengths"` tensor you separately route through
                `ctc_input_lengths` before `nn.CTCLoss` (see this
                argument's relationship to that one in the module
                docstring; they are NOT the same adjustment).
                **REQUIRED -- passing `None` raises (see below).** Used
                for two things, in this order:
                  1. IMMEDIATELY AFTER `DayLayers`, EVERY timestep at or
                     past `input_lengths[i]` is zeroed for item `i`,
                     BEFORE `_smooth` runs. This is the fix for the
                     verified batch-composition-dependence bug (see the
                     module docstring's "PADDING MUST BE MASKED BEFORE
                     SMOOTHING") -- without it, a trained `DayLayers`'
                     nonzero bias leaks into the padded region, and
                     `_smooth`'s depthwise conv then mixes that leaked
                     constant into the last few real timesteps,
                     differently depending on how much padding happened
                     to be in the batch.
                  2. After smoothing and (if `patch_size`/`patch_stride`
                     are not both 1) patching, `input_lengths` is routed
                     through `ctc_input_lengths` to get PATCHED per-item
                     lengths, and the GRU is run through
                     `nn.utils.rnn.pack_padded_sequence` / `pad_packed_
                     sequence` using THOSE (with `enforce_sorted=False`,
                     so callers need not pre-sort the batch by length,
                     and `total_length` set to the post-patching time
                     dimension so the unpacked output's time axis still
                     matches) so padded timesteps/patches never enter the
                     recurrence -- both for correctness (bidirectional,
                     if ever re-enabled) and performance (cuDNN's
                     packed-sequence path skips computation on padding
                     entirely).

        Returns:
            `log_probs`: `(time, batch, n_classes)` float32 --
            TIME-FIRST, matching `nn.CTCLoss`'s required layout directly.
            `log_softmax` has already been applied over the class
            (last) dimension exactly once; callers MUST NOT transpose
            this or apply `log_softmax` again (see the module docstring).

        Raises:
            ValueError: If `features.shape[-1] != self.n_features`,
                naming both the expected and the received shape -- a
                transposed or otherwise mis-shaped input would otherwise
                proceed silently (e.g. a `(batch, 512, time)` tensor
                passed by mistake produces no error, just wrong output).
                Also if `input_lengths is None` -- silent
                batch-composition-dependence (see above) is worse than an
                explicit failure, and every real call site
                (`training/loop.py`) has `collate_trials`'s
                `"input_lengths"` available, so there is no legitimate
                caller this refusal inconveniences.
        """
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"GRUCTC.forward: expected features.shape[-1] == "
                f"{self.n_features}, got {features.shape[-1]} "
                f"(full shape: {tuple(features.shape)}). A transposed "
                f"input (e.g. (batch, n_features, time) instead of "
                f"(batch, time, n_features)) would otherwise proceed "
                f"silently."
            )
        if input_lengths is None:
            raise ValueError(
                "GRUCTC.forward: input_lengths is required (not "
                "optional) -- it is used to mask padded timesteps to "
                "zero immediately after DayLayers and before smoothing "
                "(see the module docstring's 'PADDING MUST BE MASKED "
                "BEFORE SMOOTHING'). Skipping this silently reintroduces "
                "batch-composition-dependent outputs once DayLayers has "
                "been trained; every real call site has collate_trials's "
                "'input_lengths' available."
            )

        x = features

        if self._input_dropout is not None:
            x_bft = x.transpose(1, 2)  # (batch, n_features, time)
            x_bft = self._input_dropout(x_bft)
            x = x_bft.transpose(1, 2)  # (batch, time, n_features)

        if self.day_layers is not None:
            x = self.day_layers(x, session_indices)

        # THE FIX: zero every timestep >= input_lengths[i], for every
        # batch item i, BEFORE _smooth runs -- see the module docstring.
        # time_indices[t] < input_lengths[i] is True exactly for i's real
        # (unpadded) timesteps; broadcasting against (batch, time, 1)
        # zeroes every feature channel at every padded position.
        time_indices = torch.arange(x.size(1), device=x.device)
        valid_mask = time_indices.unsqueeze(0) < input_lengths.to(x.device).unsqueeze(1)
        x = x * valid_mask.unsqueeze(-1)

        x = self._smooth(x)

        # Order matters: patching AFTER smoothing (see _patch's docstring
        # for why smoothing at 40 ms means something different once the
        # time axis has already been strided to 80 ms). No-op when
        # patch_size==1 and patch_stride==1.
        x = self._patch(x)

        # patched_lengths, not raw input_lengths: after patching, the GRU
        # must be told each item's length in PATCHES, not original
        # timesteps -- the same formula ctc_input_lengths applies for
        # nn.CTCLoss (see the module docstring's CTC length contract),
        # computed here via the SAME method so packing and the loss can
        # never silently disagree about how long an item is.
        patched_lengths = self.ctc_input_lengths(input_lengths)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, patched_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out = self._run_gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1),
        )

        logits = self.output(gru_out)  # (batch, time, n_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, time, n_classes)
        return log_probs.transpose(0, 1)  # (time, batch, n_classes)

    def n_parameters(self) -> dict[str, int]:
        """Count ALL parameters by component, regardless of `requires_grad`.

        NOTE: despite what an earlier version of this docstring implied,
        this counts EVERY parameter unconditionally, including any a
        caller has frozen. Use `n_trainable_parameters()` for a
        `requires_grad`-filtered count -- see its docstring for an
        important caveat about `DayLayers`' hook-based freezing not being
        visible to `requires_grad` at all.

        Returns:
            A dict with keys `"day_layers"`, `"gru"`, `"output"`, and
            `"total"` (the sum of the other three). `"day_layers"` is 0
            when `use_day_layers=False` (there is no `DayLayers` module
            to count). The fixed Gaussian smoothing kernel is a buffer,
            not a parameter, and is never counted here (see its
            registration in `__init__`).
        """
        day_layers = (
            sum(p.numel() for p in self.day_layers.parameters())
            if self.day_layers is not None
            else 0
        )
        gru = sum(p.numel() for p in self.gru.parameters())
        output = sum(p.numel() for p in self.output.parameters())
        return {
            "day_layers": day_layers,
            "gru": gru,
            "output": output,
            "total": day_layers + gru + output,
        }

    def n_trainable_parameters(self) -> dict[str, int]:
        """Count parameters with `requires_grad=True`, by component.

        CAVEAT: `DayLayers.freeze_all_except` does NOT set
        `requires_grad=False` on the frozen slices -- it can't,
        `requires_grad` is a per-TENSOR flag and a session's slice is
        part of a larger shared tensor (`day_layers.py`'s module
        docstring). It masks gradients via a hook instead. This method
        therefore still reports the FULL `day_layers` parameter count as
        "trainable" even while `freeze_all_except` is active for one
        session -- it reflects `requires_grad` only, which is a real but
        incomplete picture of what's actually receiving gradient updates
        under `DayLayers`' freezing mechanism. There is no general way to
        introspect "how many of this tensor's elements currently receive
        nonzero gradient" from `requires_grad` alone.

        Returns:
            Same keys as `n_parameters()`, but each count includes only
            parameters where `p.requires_grad` is True at the moment of
            the call -- a snapshot, not a live-updating value.
        """
        day_layers = (
            sum(p.numel() for p in self.day_layers.parameters() if p.requires_grad)
            if self.day_layers is not None
            else 0
        )
        gru = sum(p.numel() for p in self.gru.parameters() if p.requires_grad)
        output = sum(p.numel() for p in self.output.parameters() if p.requires_grad)
        return {
            "day_layers": day_layers,
            "gru": gru,
            "output": output,
            "total": day_layers + gru + output,
        }
