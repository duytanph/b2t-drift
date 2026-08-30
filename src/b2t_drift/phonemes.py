"""Phoneme vocabulary and label-index conventions.

Source: Brain-to-Text '25 Kaggle competition data page
(https://www.kaggle.com/competitions/brain-to-text-25/data).

Index 0 is the CTC blank symbol; it never appears in `seq_class_ids` within
`seq_len` (see GLOSSARY.md "Dataset facts"). Indices 1-39 are the 39
standard ARPABET phonemes, in alphabetical order. Index 40 is a silence
token, not a phoneme. `seq_class_ids` values are observed empirically to
fall in the range 1-40.
"""

from __future__ import annotations

from collections.abc import Sequence

LOGIT_TO_PHONEME = [
    'BLANK',    # "BLANK" = CTC blank symbol
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',    # "|" = silence token
]

assert len(LOGIT_TO_PHONEME) == 41

BLANK_INDEX = 0
SILENCE_INDEX = 40
N_CLASSES = len(LOGIT_TO_PHONEME)  # 41

PHONEME_TO_LOGIT = {phoneme: index for index, phoneme in enumerate(LOGIT_TO_PHONEME)}


def decode_ids(ids: Sequence[int]) -> list[str]:
    """Map integer label IDs to phoneme strings.

    Args:
        ids: Label indices, each expected in `[0, N_CLASSES)`.

    Returns:
        The corresponding phoneme strings, in order.

    Raises:
        ValueError: If any id is outside `[0, N_CLASSES)`. Out-of-range
            values are never silently skipped or clamped.
    """
    decoded = []
    for i in ids:
        if not (0 <= i < N_CLASSES):
            raise ValueError(f"label id {i} out of range [0, {N_CLASSES})")
        decoded.append(LOGIT_TO_PHONEME[i])
    return decoded
