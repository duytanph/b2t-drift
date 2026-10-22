"""Measure padding waste for random vs. length-bucketed batching.

Computes, for a given batch size, the fraction of each batch's time
dimension spent on padding rather than real data:

    padding waste = 1 - (sum of true lengths) / (batch_size * sum of batch maxima)

for (a) naive random batching and (b) the real `LengthBucketedSampler`
actually used by `scripts/02_train.py` (`src/b2t_drift/data/collate.py`),
over the true per-trial `n_time_steps` distribution read directly from
`data/raw/hdf5_data_final`'s HDF5 attributes (the committed
`data/interim/session_manifest.csv` only has per-session aggregates, not
the per-trial lengths this measurement needs). Reads attributes only --
no feature arrays, no GPU, no checkpoint -- so it runs anywhere in seconds.

This exists because a prior version of this measurement (35.5% / 13.7%,
then a since-superseded 47.6% / 9.5%) was quoted in `docs/DESIGN.md` with
no reproducible source. See `docs/FINDINGS.md`: "Length-distribution
documentation errors traced to one session's statistics; padding waste
re-measured" for the investigation this script resulted from.

Run with: uv run tools/measure_padding_waste.py
"""

from __future__ import annotations

import argparse
import random
import statistics as st
import sys
from pathlib import Path

from b2t_drift.data.collate import LengthBucketedSampler
from b2t_drift.data.hdf5_io import discover_sessions, iter_trial_ids

DEFAULT_DATA_ROOT = Path("data/raw/hdf5_data_final")
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEEDS = 50
# Matches scripts/02_train.py's train-partition DataLoader (the only one
# of the two this sampler is actually used with at full scale).
DEFAULT_DROP_LAST = True


def padding_waste(batches: list[list[int]], lengths: list[int], batch_size: int) -> float:
    """Fraction of total batch capacity spent on padding, not real data."""
    total_true = sum(lengths[i] for batch in batches for i in batch)
    total_capacity = sum(batch_size * max(lengths[i] for i in batch) for batch in batches)
    return 1.0 - total_true / total_capacity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Directory containing per-session HDF5 subdirectories (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Trials per batch, i.e. the 'pool' size (default: %(default)s; "
        "this codebase has no separate pool concept -- LengthBucketedSampler "
        "takes only batch_size).",
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help="Number of random shuffles/seeds to average over (default: %(default)s).",
    )
    parser.add_argument(
        "--drop-last", action=argparse.BooleanOptionalAction, default=DEFAULT_DROP_LAST,
        help="Drop a final short batch, matching scripts/02_train.py's train "
        "DataLoader (default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        sys.exit(f"Data root not found: {args.data_root}")

    sessions = discover_sessions(args.data_root)
    trial_ids = []
    for s in sessions:
        trial_ids.extend(iter_trial_ids(args.data_root, s.session_id))
    lengths = [t.n_time_steps for t in trial_ids]
    n = len(lengths)

    random_waste = []
    bucketed_waste = []
    for seed in range(args.seeds):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        batches = [order[i : i + args.batch_size] for i in range(0, n, args.batch_size)]
        if args.drop_last and batches and len(batches[-1]) < args.batch_size:
            batches = batches[:-1]
        random_waste.append(padding_waste(batches, lengths, args.batch_size))

        sampler = LengthBucketedSampler(
            trial_ids, batch_size=args.batch_size, shuffle=True, seed=seed,
            drop_last=args.drop_last,
        )
        sampler.set_epoch(0)
        bucketed_waste.append(padding_waste(list(sampler), lengths, args.batch_size))

    print(f"{n} labeled trials, batch_size={args.batch_size}, "
          f"drop_last={args.drop_last}, {args.seeds} shuffles")
    print(f"random batching:   {st.mean(random_waste)*100:.1f}% padding "
          f"(std {st.stdev(random_waste)*100:.2f}%)")
    print(f"LengthBucketedSampler: {st.mean(bucketed_waste)*100:.1f}% padding "
          f"(std {st.stdev(bucketed_waste)*100:.2f}%)")


if __name__ == "__main__":
    main()
