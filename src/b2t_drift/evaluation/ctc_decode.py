"""CTC decoding: turning model logits into phoneme sequences.

Responsible for greedy (argmax + collapse-repeats + remove-blanks) decoding
of CTC logits. Beam search is out of scope unless later needed.

THE SILENCE-TOKEN DECISION (also recorded in `docs/DESIGN.md` section 9):
CTC blanks are removed by `greedy_ctc_decode`, and consecutive repeats are
collapsed, per the standard CTC decoding procedure -- but SILENCE (class
40) is NOT removed. It is scored as a regular token, exactly like any of
the 39 phonemes, in whatever downstream comparison (e.g. Levenshtein
distance for PER) consumes this function's output. This matches the
Brain-to-Text lineage's field convention: a 41-token vocabulary (39
phonemes + silence + CTC blank), decoded by removing repeats and blanks
and computing edit distance against the ground-truth phoneme sequence
(which itself includes silence tokens), normalized by the total number of
ground-truth phonemes. The reason silence stays in: it marks word
boundaries, and a decoder that gets every phoneme right but misplaces
those boundaries produces text a downstream language model can't use
(`docs/DATASET.md` section 15) -- since the pipeline this project measures
depends on those boundaries, they belong in the metric, not just in the
raw decode.
"""

from __future__ import annotations

import torch


def greedy_ctc_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    blank_index: int = 0,
    n_classes: int = 41,
) -> list[list[int]]:
    """Greedily decode a batch of CTC logits into label-index sequences.

    Collapses repeated consecutive predictions and removes blank tokens
    (but NOT the silence token -- see the module docstring), per the
    standard CTC greedy decoding procedure. Works identically whether
    `logits` are raw logits or log-probabilities (e.g. `GRUCTC.forward`'s
    output, transposed to batch-first) -- `argmax` is invariant under
    `log_softmax`, which is a monotonic, per-row transform.

    THE SILENT AXIS-ORDER FAILURE THIS FUNCTION USED TO ALLOW: `logits`
    must be `(batch, time, vocab)`, but `GRUCTC.forward` returns
    `(time, batch, n_classes)` -- passing the latter by mistake used to
    NOT raise: `argmax(dim=-1)` works on either axis order, and reading
    `time` as `batch` just produced `max_time` garbage-length sequences
    that a downstream `zip(decoded, references)` would silently truncate
    against, producing a plausible-looking PER computed entirely from
    noise. The assertion on `logits.shape[0] == lengths.shape[0]` below
    is what actually catches this: a genuine batch-first tensor's leading
    dimension matches the batch-sized `lengths` argument; a mistakenly
    time-first tensor's leading dimension (the sequence length) essentially
    never does.

    Args:
        logits: Per-timestep class scores, shape `(batch, time,
            vocab_size)`. NOTE: this is BATCH-FIRST, the opposite of
            `GRUCTC.forward`'s time-first return -- callers passing a
            model's output here must `.transpose(0, 1)` it first.
        lengths: `(batch,)`, each item's true (unpadded) time length.
            REQUIRED (not optional): decoding a padded region has no
            defined "correct" prediction, and appends spurious trailing
            symbols to the decoded sequence -- an opt-out default here
            would eventually be forgotten at some call site and fire
            silently, exactly the kind of bug this module's docstring
            warns about. Timesteps at or past `lengths[i]` are excluded
            from item `i`'s decode before collapsing.
        blank_index: Index of the CTC blank token. Removed from the
            decoded sequence (silence, a different, non-blank class
            index, is not -- see the module docstring).
        n_classes: Expected size of `logits`' class dimension
            (`logits.shape[-1]`), checked by the assertion below. Default
            41 matches this project's fixed CTC vocabulary
            (`b2t_drift.phonemes.N_CLASSES`) but is a parameter, not a
            hardcoded constant, so this function isn't silently wrong if
            ever reused with a different vocabulary size.

    Returns:
        One decoded label-index sequence per batch element, with
        consecutive repeats collapsed and `blank_index` removed (silence
        retained).

    Raises:
        AssertionError: if `logits.ndim != 3`; if
            `logits.shape[0] != lengths.shape[0]` (the check that
            actually catches the axis-order failure described above,
            naming both shapes); or if `logits.shape[-1] != n_classes`
            (naming both).
    """
    assert logits.ndim == 3, (
        f"greedy_ctc_decode: expected logits.ndim == 3 (batch, time, "
        f"vocab_size), got shape {tuple(logits.shape)}."
    )
    assert logits.shape[0] == lengths.shape[0], (
        f"greedy_ctc_decode: logits.shape[0]={logits.shape[0]} does not "
        f"match lengths.shape[0]={lengths.shape[0]} -- if logits came "
        f"from GRUCTC.forward (time, batch, n_classes), you must "
        f".transpose(0, 1) it before calling this function (see the "
        f"module docstring's 'SILENT AXIS-ORDER FAILURE' note)."
    )
    assert logits.shape[-1] == n_classes, (
        f"greedy_ctc_decode: logits.shape[-1]={logits.shape[-1]} does "
        f"not match n_classes={n_classes}."
    )

    predictions = logits.argmax(dim=-1)  # (batch, time)
    batch_size = predictions.shape[0]

    decoded: list[list[int]] = []
    for b in range(batch_size):
        t = int(lengths[b])
        seq = predictions[b, :t].tolist()

        collapsed = []
        prev = None
        for label in seq:
            if label != prev:
                collapsed.append(label)
            prev = label
        decoded.append([label for label in collapsed if label != blank_index])

    return decoded
