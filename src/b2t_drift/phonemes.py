"""Phoneme vocabulary and label-index conventions.

Responsible for the canonical mapping between phoneme symbols and the
integer class indices used by the CTC loss and decoder, including the blank
token convention. Do not hardcode a specific vocabulary here until the T15
label format has been inspected.
"""

from __future__ import annotations


class PhonemeVocabulary:
    """Bidirectional mapping between phoneme symbols and label indices.

    TODO: confirm against the actual T15 label files whether the blank
    token is index 0, and whether silence/noise symbols need special
    handling.
    """

    def __init__(self, symbols: list[str], blank_index: int = 0) -> None:
        """Build a vocabulary from an ordered list of phoneme symbols.

        Args:
            symbols: Phoneme symbols, in index order (excluding blank).
            blank_index: Index reserved for the CTC blank token.
        """
        raise NotImplementedError

    def encode(self, phonemes: list[str]) -> list[int]:
        """Convert a sequence of phoneme symbols to label indices.

        Args:
            phonemes: Phoneme symbols.

        Returns:
            Corresponding label indices.
        """
        raise NotImplementedError

    def decode(self, indices: list[int]) -> list[str]:
        """Convert a sequence of label indices back to phoneme symbols.

        Args:
            indices: Label indices.

        Returns:
            Corresponding phoneme symbols.
        """
        raise NotImplementedError

    @property
    def size(self) -> int:
        """Total vocabulary size, including the blank token."""
        raise NotImplementedError
