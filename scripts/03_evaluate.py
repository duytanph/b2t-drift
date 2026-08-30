"""CLI: evaluate a trained checkpoint on held-out sessions.

Thin wrapper only -- calls into `b2t_drift.training.checkpoint` and
`b2t_drift.evaluation`. No logic belongs here.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point: load a checkpoint and report evaluation metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
