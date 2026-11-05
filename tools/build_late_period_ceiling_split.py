"""Build splits/late_period_ceiling_switchboard.json.

docs/FINDINGS.md: "Late-period ceiling measured directly: the signal
itself, not the recalibration method, limits the +597d result"
(2026-11-04). Measures how well ANY decoder can do on a late session's
signal with drift removed from the problem: train AND val both come from
the same six late Switchboard sessions (+504d to +597d from the s05 cut),
so there is no elapsed-time gap -- this isolates signal quality from
drift.

Pool: the six Switchboard sessions past +500d from the s05 cut
(t15.2025.01.10/.01.12/.03.14/.03.16/.03.30/.04.13 -- the specific set is
itself the deliberate design decision this measurement made, not a
general parameterization, per `docs/FINDINGS.md`, same entry), their
labeled (train+val partition) Switchboard trials only -- 836 trials, 92%
of s05's 908 training trials, a fair-size comparison.

Split: per-session 90/10 (val_fraction, same convention and seed as
build_temporal_split), THEN sentence-level dedup on top (the
recalibration grid's stricter rule, not build_temporal_split's trial-only
default): any val trial whose sentence text also appears anywhere in
train is dropped.

Manifest kind="official" (not "temporal"): bypasses validate_split's
cut-date / eval_sessions checks, which assume a train-precedes-eval
guarantee this manifest deliberately does NOT have.

No CUDA needed -- this only reads HDF5 attributes and writes a JSON
manifest. Training on the manifest this writes
(`uv run scripts/02_train.py --config configs/experiment/
switchboard_baseline.yaml --split splits/late_period_ceiling_switchboard.json
--run-name late_period_ceiling`) does require CUDA; see
tools/evaluate_late_period_ceiling.py for scoring the result.

Run with: uv run tools/build_late_period_ceiling_split.py
"""

from __future__ import annotations

import argparse
import random
from datetime import UTC, datetime
from pathlib import Path

from b2t_drift.config import load_yaml
from b2t_drift.data.hdf5_io import discover_sessions, iter_trial_ids
from b2t_drift.data.splits import load_corpus_labels, validate_split, write_split
from b2t_drift.evaluation.calibration import normalize_sentence
from b2t_drift.utils.provenance import get_git_commit_hash

LATE_SESSIONS = [
    "t15.2025.01.10",
    "t15.2025.01.12",
    "t15.2025.03.14",
    "t15.2025.03.16",
    "t15.2025.03.30",
    "t15.2025.04.13",
]
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SEED = 0
CORPUS_FILTER = "Switchboard"
DEFAULT_OUT_PATH = Path("splits/late_period_ceiling_switchboard.json")
DEFAULT_DATA_CONFIG = Path("configs/data/t15.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config", type=Path, default=DEFAULT_DATA_CONFIG,
        help="Path to the data config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
        help="Per-session fraction held out for val, before sentence dedup (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed for the per-session val-trial sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_PATH,
        help="Output manifest path (default: %(default)s).",
    )
    args = parser.parse_args()

    data_config = load_yaml(args.data_config)
    data_root = Path(data_config["raw_data_dir"])
    corpus_csv = Path(data_config["corpus_csv_path"])

    sessions = discover_sessions(data_root)
    session_to_index = {s.session_id: i for i, s in enumerate(sessions)}
    corpus_labels = load_corpus_labels(corpus_csv)

    rng = random.Random(args.seed)
    train_refs: list[list[str]] = []
    val_refs: list[list[str]] = []
    train_sentences: set[str] = set()
    pool_summary = []

    per_session_trials = {}
    for session_id in LATE_SESSIONS:
        trials = [
            t for t in iter_trial_ids(data_root, session_id, partitions=("train", "val"))
            if corpus_labels.get((t.session_id, t.block_num)) == CORPUS_FILTER
        ]
        per_session_trials[session_id] = trials
        n = len(trials)
        n_val = min(n, max(1, round(args.val_fraction * n))) if n else 0
        val_indices = set(rng.sample(range(n), n_val)) if n_val else set()
        for i, t in enumerate(trials):
            ref = [t.session_id, t.partition, t.trial_key]
            if i in val_indices:
                val_refs.append(ref)
            else:
                train_refs.append(ref)
                train_sentences.add(normalize_sentence(t.sentence_label))
        pool_summary.append((session_id, n, n_val))

    # Sentence-level dedup: drop any val trial whose sentence also appears
    # anywhere in train (recalibration grid's rule, applied on top of the
    # per-session split above -- build_temporal_split's own default is
    # trial-level only).
    sentence_lookup = {}
    for session_id in LATE_SESSIONS:
        for t in per_session_trials[session_id]:
            sentence_lookup[(t.session_id, t.partition, t.trial_key)] = t

    kept_val_refs = []
    dropped_val_refs = []
    for ref in val_refs:
        t = sentence_lookup[tuple(ref)]
        if normalize_sentence(t.sentence_label) in train_sentences:
            dropped_val_refs.append(ref)
        else:
            kept_val_refs.append(ref)

    print("=== Late-period pool composition ===")
    total = 0
    for session_id, n, n_val in pool_summary:
        print(f"  {session_id}: {n} labeled Switchboard trials, {n_val} assigned to val (pre-dedup)")
        total += n
    print(f"  Pooled total: {total} trials (s05 training block: 908, ratio {total / 908:.3f})")
    print(f"  val (pre-dedup): {len(val_refs)}, dropped to sentence overlap: {len(dropped_val_refs)}, "
          f"val (post-dedup): {len(kept_val_refs)}")
    print(f"  train: {len(train_refs)}")

    manifest = {
        "name": "late_period_ceiling_switchboard",
        "kind": "official",  # see module docstring: bypasses the temporal/cut-date checks
        "created_by": "tools/build_late_period_ceiling_split.py",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_hash(),
        "corpus_filter": CORPUS_FILTER,
        "late_sessions": LATE_SESSIONS,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "session_to_index": session_to_index,
        "train": train_refs,
        "val": kept_val_refs,
        "eval_sessions": [],
    }
    validate_split(manifest, corpus_labels=corpus_labels, data_root=data_root)
    write_split(manifest, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
