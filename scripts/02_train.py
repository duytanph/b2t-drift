"""CLI: train a model from an experiment config.

Thin wrapper only -- parses a config path, then calls into
`b2t_drift.config`, `b2t_drift.data`, `b2t_drift.models`, and
`b2t_drift.training`. No logic belongs here.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point: load an experiment config and run training."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
