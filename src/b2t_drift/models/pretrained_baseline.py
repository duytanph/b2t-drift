"""Reimplementation of, and checkpoint adapter for, the officially released
T15 pretrained RNN baseline (`data/external/t15_pretrained_rnn_baseline/`).

WHY THIS EXISTS, SEPARATELY FROM `GRUCTC`. This checkpoint was trained on
train partitions from all 45 sessions (`docs/DESIGN.md` section 2), so it
has no held-out future and cannot measure drift -- but for exactly that
reason it is a candidate ALIGNMENT TOOL for Phase 2's tuning analysis: its
quality should not trend with session date, unlike our own drift-curve
decoders. This module exists only to load and run that checkpoint, never to
train one -- see `docs/DESIGN.md` sections 2-3 and `CLAUDE.md`'s "no
premature abstraction" for why this stays a second, narrow class rather
than a generalization of `GRUCTC` itself.

ARCHITECTURE DIFFERENCES FROM `GRUCTC`, gathered here. `PretrainedBaselineRNN`
subclasses `GRUCTC` and passes hyperparameters matching the checkpoint
(`args.yaml`): `n_layers=5` (ours: 3), `hidden_size=768` (ours: 512),
`patch_size=14, patch_stride=4` (ours, by default: 1/1 -- no patching),
`day_layer_parameterization="full"` (ours, by default: "diagonal"),
`n_classes=41` (fixed by data, same as ours). These are all `GRUCTC`
constructor arguments already -- no subclassing needed for them.

ONE ARCHITECTURE ELEMENT `GRUCTC` HAS NO EQUIVALENT FOR: `h0`, a single
LEARNED (1, 1, hidden_size) initial hidden state, broadcast across every
GRU layer and every batch item -- discovered by this module's adapter, not
previously reported (the original checkpoint inspection,
`docs/FINDINGS.md`: "Block count decreases over the recording...", did not
find it). `GRUCTC` always uses `nn.GRU`'s default zero-initialized hidden
state. `PretrainedBaselineRNN` adds `self.h0` and overrides
`GRUCTC._run_gru` (a method `GRUCTC` exposes for exactly this purpose) to
pass it, expanded to `(n_layers, batch, hidden_size)`, into `self.gru`.
Because `h0` is IDENTICAL across the batch dimension, `nn.GRU`'s internal
permutation of `hx` by a `PackedSequence`'s `sorted_indices` (needed since
`forward` packs with `enforce_sorted=False`) has no observable effect --
permuting a matrix with identical rows returns the same matrix.

WHY DAY-LAYER APPLICATION ORDER RELATIVE TO SMOOTHING DOES NOT MATTER,
EVEN THOUGH THE OFFICIAL PIPELINE APPLIES SMOOTHING AS A DATA TRANSFORM
(`args.yaml`'s `dataset.data_transforms.smooth_data`/`smooth_kernel_std:
2`) WHILE `GRUCTC.forward` APPLIES IT AFTER `DayLayers`, INSIDE THE MODEL.
The day layer is a PER-SESSION AFFINE MAP APPLIED IDENTICALLY AT EVERY
TIMESTEP of a given trial (`y_t = x_t @ W[s] + b[s]`, same `W[s]`, `b[s]`
for every `t`); smoothing is a per-channel WEIGHTED AVERAGE ACROSS TIME
with weights summing to 1. These commute:
`smooth(x @ W + b)_t = sum_k kernel_k * (x_{t+k} @ W + b)
                     = (sum_k kernel_k * x_{t+k}) @ W + b * sum_k kernel_k
                     = smooth(x)_t @ W + b`
(using `sum_k kernel_k = 1`). So applying the day layer before smoothing
(this module, via inherited `GRUCTC.forward`) or after (the official
pipeline, as a pre-model data transform) produces the same result up to
boundary/padding effects, which `GRUCTC.forward`'s own padding mask (run
before `_smooth`) already handles. No reordering needed.

WHAT IS DELIBERATELY *NOT* REPRODUCED: `args.yaml`'s
`dataset.data_transforms.white_noise_std`, `constant_offset_std`,
`random_walk_std`, `static_gain_std`, `random_cut`, and
`input_network.input_layer_dropout` are standard TRAINING-TIME data
augmentation / regularization, inactive at inference -- this module never
trains, only evaluates, so these are correctly absent rather than an
oversight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.training.checkpoint import load_torch_checkpoint_with_numpy_fallback

N_SESSIONS = 45
N_FEATURES = 512
HIDDEN_SIZE = 768
N_LAYERS = 5
N_CLASSES = 41
PATCH_SIZE = 14
PATCH_STRIDE = 4
SMOOTHING_SIGMA_BINS = 2.0
DAY_LAYER_PARAMETERIZATION = "full"

# torch.compile wraps the original module as `_orig_mod`, prefixing every
# state_dict key -- present on every key in the released checkpoint.
_CHECKPOINT_KEY_PREFIX = "_orig_mod."


class PretrainedBaselineRNN(GRUCTC):
    """`GRUCTC`, configured to match the official checkpoint's architecture,
    plus a learned initial GRU hidden state (`h0`) `GRUCTC` has no
    equivalent for. See the module docstring for the full architecture
    comparison and why day-layer/smoothing order doesn't matter.

    Never trained in this project -- see the module docstring.
    """

    def __init__(self, n_sessions: int = N_SESSIONS) -> None:
        """Build the model with the official checkpoint's fixed hyperparameters.

        Args:
            n_sessions: Number of day-layer slices to allocate. 45 (every
                T15 session) by default, matching the checkpoint; a
                smaller value is only useful for tests.
        """
        super().__init__(
            n_sessions=n_sessions,
            n_features=N_FEATURES,
            hidden_size=HIDDEN_SIZE,
            n_layers=N_LAYERS,
            n_classes=N_CLASSES,
            dropout=0.0,
            bidirectional=False,
            use_day_layers=True,
            input_dropout=0.0,
            smoothing_sigma_bins=SMOOTHING_SIGMA_BINS,
            patch_size=PATCH_SIZE,
            patch_stride=PATCH_STRIDE,
            day_layer_parameterization=DAY_LAYER_PARAMETERIZATION,
        )
        # Zero-init here is irrelevant in practice: the adapter always
        # overwrites this from the checkpoint before inference. Zero
        # (rather than uninitialized/random) only matters for a model
        # constructed but never loaded, e.g. in a unit test.
        self.h0 = nn.Parameter(torch.zeros(1, 1, HIDDEN_SIZE))

    def _run_gru(self, packed: nn.utils.rnn.PackedSequence) -> nn.utils.rnn.PackedSequence:
        """Override: run `self.gru` with `h0` broadcast to every layer and
        batch item, instead of `GRUCTC`'s (PyTorch-default) zero hidden
        state. See the module docstring for why the batch-dimension
        broadcast survives `enforce_sorted=False`'s internal permutation
        unchanged.
        """
        batch_size = int(packed.batch_sizes[0])
        h0 = self.h0.expand(self.n_layers, batch_size, self.hidden_size).contiguous()
        packed_out, _ = self.gru(packed, h0)
        return packed_out


def build_pretrained_baseline_model(n_sessions: int = N_SESSIONS) -> PretrainedBaselineRNN:
    """Construct a fresh, randomly-initialized `PretrainedBaselineRNN`.

    Args:
        n_sessions: See `PretrainedBaselineRNN.__init__`.

    Returns:
        A model ready to have real weights loaded into it via
        `load_pretrained_baseline_checkpoint`.
    """
    return PretrainedBaselineRNN(n_sessions=n_sessions)


def load_pretrained_baseline_checkpoint(
    path: Path,
    model: PretrainedBaselineRNN,
    map_location: str = "cpu",
    transpose_day_weights: bool = False,
) -> dict[str, Any]:
    """Load the officially released checkpoint's weights into a
    `PretrainedBaselineRNN`, explicitly remapping its per-session,
    separately-keyed day-layer tensors into our stacked `DayLayers`
    representation.

    ASSERTS EVERY CHECKPOINT PARAMETER IS ACCOUNTED FOR -- no silent
    skipping. Every one of the checkpoint's 113 top-level parameters is
    either (a) consumed by an explicit mapping rule below, or (b) causes
    this function to raise, naming exactly which key(s) could not be
    mapped. There is no fallback path that silently drops a parameter.

    THE DAY-WEIGHT ORIENTATION AMBIGUITY. `DayLayers` (`"full"`
    parameterization) computes `y_t = x_t @ W[s] + b[s]`. The checkpoint's
    `day_weights.{i}` are square (512, 512) and were identity matrices
    before training (`docs/FINDINGS.md`), so shape and initial value alone
    cannot distinguish `W[s] = day_weights.i` (used directly, matching
    `DayLayers`' own convention) from `W[s] = day_weights.i.T` (an
    `nn.Linear`-style stored weight, `(out_features, in_features)`,
    requiring a transpose before `DayLayers`' `x @ W` convention applies).
    `transpose_day_weights` selects between them; the caller is expected
    to determine the correct value empirically (does inference reproduce
    the checkpoint's recorded `val_PER` -- see
    `scratch/verify_pretrained_checkpoint.py`), not guess it here.

    Args:
        path: Path to the checkpoint file (`best_checkpoint`).
        model: A `PretrainedBaselineRNN` to load weights into, in place.
        map_location: Passed to `torch.load`.
        transpose_day_weights: If True, transpose each `day_weights.{i}`
            matrix before stacking into `DayLayers.weights`. See above.

    Returns:
        A dict with keys `"val_PER"`, `"val_loss"` (the checkpoint's own
        recorded validation metrics, for comparison against a fresh
        evaluation), `"n_checkpoint_params"` (top-level parameter count,
        113 for the released checkpoint), `"n_mapped_params"` (must equal
        `n_checkpoint_params` on success), and `"transpose_day_weights"`
        (echoes the argument, so a report can state which convention was
        used without the caller tracking it separately).

    Raises:
        ValueError: If any checkpoint parameter cannot be mapped to a
            destination in `model`'s state dict (names every such key), if
            an expected checkpoint key is missing (names it), or if a
            mapped destination key does not exist in `model.state_dict()`
            (names it) -- any of these stops loading rather than
            proceeding with a partial or guessed state.
    """
    checkpoint = load_torch_checkpoint_with_numpy_fallback(
        path, map_location=map_location, caller="load_pretrained_baseline_checkpoint",
    )
    raw_sd = checkpoint["model_state_dict"]

    stripped: dict[str, torch.Tensor] = {}
    for key, value in raw_sd.items():
        if not key.startswith(_CHECKPOINT_KEY_PREFIX):
            raise ValueError(
                f"load_pretrained_baseline_checkpoint: checkpoint key "
                f"{key!r} does not start with the expected "
                f"{_CHECKPOINT_KEY_PREFIX!r} (torch.compile) prefix -- "
                f"the checkpoint's format may have changed."
            )
        stripped[key[len(_CHECKPOINT_KEY_PREFIX):]] = value

    n_sessions = model.n_sessions
    consumed: set[str] = set()
    mapped: dict[str, torch.Tensor] = {}

    # --- Day layers: 45 separately-keyed (weight, bias) pairs -> one
    # stacked (n_sessions, 512, 512) / (n_sessions, 512) pair, matching
    # DayLayers' "full" parameterization storage exactly.
    day_weights: list[torch.Tensor] = []
    day_biases: list[torch.Tensor] = []
    for i in range(n_sessions):
        weight_key, bias_key = f"day_weights.{i}", f"day_biases.{i}"
        missing = [k for k in (weight_key, bias_key) if k not in stripped]
        if missing:
            raise ValueError(
                f"load_pretrained_baseline_checkpoint: missing day-layer "
                f"key(s) for session index {i}: {missing}."
            )
        w = stripped[weight_key]
        day_weights.append(w.T if transpose_day_weights else w)
        day_biases.append(stripped[bias_key].squeeze(0))
        consumed.add(weight_key)
        consumed.add(bias_key)
    mapped["day_layers.weights"] = torch.stack(day_weights, dim=0)
    mapped["day_layers.biases"] = torch.stack(day_biases, dim=0)

    # --- GRU: direct copy, same names, same shapes (hidden_size=768,
    # n_layers=5, patched 7,168-dim input all set at construction time).
    for layer in range(model.n_layers):
        for part in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            key = f"gru.{part}_l{layer}"
            if key not in stripped:
                raise ValueError(
                    f"load_pretrained_baseline_checkpoint: missing GRU key {key!r}."
                )
            mapped[f"gru.{part}_l{layer}"] = stripped[key]
            consumed.add(key)

    # --- Output projection: checkpoint calls it "out", GRUCTC calls it
    # "output" -- a pure rename, no reshape.
    for part in ("weight", "bias"):
        key = f"out.{part}"
        if key not in stripped:
            raise ValueError(
                f"load_pretrained_baseline_checkpoint: missing output key {key!r}."
            )
        mapped[f"output.{part}"] = stripped[key]
        consumed.add(key)

    # --- h0: the learned initial hidden state PretrainedBaselineRNN adds
    # specifically to hold this (see the module docstring). A direct copy
    # -- shapes match exactly ((1, 1, 768)).
    if "h0" not in stripped:
        raise ValueError(
            "load_pretrained_baseline_checkpoint: checkpoint has no 'h0' "
            "key -- expected the published architecture's learned initial "
            "GRU hidden state (see the module docstring)."
        )
    mapped["h0"] = stripped["h0"]
    consumed.add("h0")

    unmapped = set(stripped) - consumed
    if unmapped:
        raise ValueError(
            f"load_pretrained_baseline_checkpoint: {len(unmapped)} "
            f"checkpoint parameter(s) have no mapping rule and were NOT "
            f"loaded (no silent skipping): {sorted(unmapped)}."
        )

    target_state_dict = model.state_dict()
    unknown_destinations = set(mapped) - set(target_state_dict)
    if unknown_destinations:
        raise ValueError(
            f"load_pretrained_baseline_checkpoint: mapped key(s) "
            f"{sorted(unknown_destinations)} do not exist in "
            f"model.state_dict() -- PretrainedBaselineRNN's architecture "
            f"does not match what this adapter assumes."
        )
    target_state_dict.update(mapped)
    model.load_state_dict(target_state_dict, strict=True)

    return {
        "val_PER": checkpoint.get("val_PER"),
        "val_loss": checkpoint.get("val_loss"),
        "n_checkpoint_params": len(stripped),
        "n_mapped_params": len(consumed),
        "transpose_day_weights": transpose_day_weights,
    }
