"""CLI: build and write the temporal train/val/test session split manifest.

Thin wrapper only -- calls into `b2t_drift.data.splits`. No logic belongs
here.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point: compute temporal splits and write them to `splits/`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
