"""CLI: produce the session-vs-error-rate drift curve figure.

Thin wrapper only -- calls into `b2t_drift.evaluation.metrics` and plotting
helpers. No logic belongs here.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point: compute per-session error rates and plot drift over time."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
