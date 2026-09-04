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
  3. (optional) Gaussian temporal smoothing: fixed, non-learned depthwise
     `conv1d` over the time axis. See `_smooth`.
  4. `nn.GRU`: the shared recurrent encoder. Uses PyTorch's built-in,
     cuDNN-backed implementation directly -- NOT a hand-rolled per-timestep
     loop, which would be 10-50x slower on GPU because it can't use
     cuDNN's fused RNN kernels. Do not "unroll" this for debuggability;
     use `torch.autograd` / a debugger on the fused op instead.
  5. A linear projection to `n_classes` logits, then `log_softmax` over the
     class dimension.

THE CTC LENGTH CONTRACT (see also `data/collate.py`'s module docstring,
which this model's caller must also have read). This version applies NO
temporal downsampling anywhere in the forward pass -- the GRU consumes and
emits one output per input timestep, 1:1. `input_lengths` therefore pass
through `ctc_input_lengths` UNCHANGED. This is a deliberate, explicit
scope decision, not an oversight: the published baseline concatenates 14
consecutive 20 ms bins at stride 4 before its GRU (a 7,168-dim input at an
effective 80 ms output stride), which WOULD require dividing
`input_lengths` by 4 before `nn.CTCLoss` (`CLAUDE.md`'s "known pitfalls" --
getting this wrong is the single most likely place for a silent bug in
this whole pipeline, since a wrong length doesn't raise, it just corrupts
the loss). That comparison is future work, not inherited here by default.
`ctc_input_lengths` exists specifically so that when downsampling IS added
later, there is exactly one place in this file that has to change, and
every call site that already routes lengths through it picks up the fix
automatically instead of needing to be found and edited individually.

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

DEVIATIONS FROM THE PUBLISHED BASELINE, gathered in one place:
  - Plain 512-dim input at 20 ms bins, no 14-bin/stride-4 concatenation
    (see the CTC length contract note above).
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
    contract, and the time-first output convention.
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
        """
        super().__init__()
        self.n_sessions = n_sessions
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.use_day_layers = use_day_layers
        self.input_dropout = input_dropout
        self.smoothing_sigma_bins = smoothing_sigma_bins

        self._input_dropout = nn.Dropout1d(input_dropout) if input_dropout > 0 else None

        self.day_layers = DayLayers(n_sessions, n_features) if use_day_layers else None

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
            self._smoothing_kernel = None

        self.gru = nn.GRU(
            input_size=n_features,
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

    def ctc_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map raw per-trial `input_lengths` to CTC-ready lengths.

        Currently the IDENTITY function: this version of `GRUCTC` applies
        no temporal downsampling anywhere (see the module docstring's CTC
        length contract), so `nn.CTCLoss`'s `input_lengths` argument is
        exactly the same as `collate_trials`'s `input_lengths`.

        THIS IS THE SINGLE PLACE THAT MUST CHANGE if temporal downsampling
        (striding, pooling, patching the time axis) is ever added to this
        model. Callers should ALWAYS route `input_lengths` through this
        method before passing them to `nn.CTCLoss`, even though today it
        does nothing -- that discipline is what prevents the length
        adjustment from being silently forgotten at whichever call site
        adds the downsampling later (`CLAUDE.md`'s "known pitfalls").

        Args:
            input_lengths: `(batch,)` int64, from `collate_trials`.

        Returns:
            `(batch,)` int64. Currently `input_lengths` unchanged
            (the same tensor object, not a copy -- there is nothing to
            compute yet).
        """
        return input_lengths

    def forward(self, features: torch.Tensor, session_indices: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep log-class-probabilities.

        Args:
            features: `(batch, time, 512)` float32, from
                `collate_trials`'s `"features"`.
            session_indices: `(batch,)` int64, from `collate_trials`'s
                `"session_indices"`. Read even when `use_day_layers=False`
                (accepted for a uniform call signature) but unused in
                that case.

        Returns:
            `log_probs`: `(time, batch, n_classes)` float32 --
            TIME-FIRST, matching `nn.CTCLoss`'s required layout directly.
            `log_softmax` has already been applied over the class
            (last) dimension exactly once; callers MUST NOT transpose
            this or apply `log_softmax` again (see the module docstring).
        """
        x = features

        if self._input_dropout is not None:
            x_bft = x.transpose(1, 2)  # (batch, n_features, time)
            x_bft = self._input_dropout(x_bft)
            x = x_bft.transpose(1, 2)  # (batch, time, n_features)

        if self.day_layers is not None:
            x = self.day_layers(x, session_indices)

        x = self._smooth(x)

        gru_out, _ = self.gru(x)  # (batch, time, hidden_size * n_directions)
        logits = self.output(gru_out)  # (batch, time, n_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, time, n_classes)
        return log_probs.transpose(0, 1)  # (time, batch, n_classes)

    def n_parameters(self) -> dict[str, int]:
        """Count trainable parameters by component.

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
