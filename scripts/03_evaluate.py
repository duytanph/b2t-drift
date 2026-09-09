"""CLI: evaluate a trained checkpoint on every held-out session of a split.

Thin wrapper only -- loads a checkpoint and a split manifest, builds one
dataloader per held-out session, calls library code in `b2t_drift.training`
/ `b2t_drift.evaluation` to score it, and writes the results. No logic
belongs here (`CLAUDE.md`): the model, the decode/alignment machinery, and
the per-session PER/CI aggregation all live in `src/b2t_drift/`.

THE ARM 0 (NO-ADAPTATION) PATH. Every held-out session indexes into the
checkpoint's `DayLayers` via `session_to_index` FROM THE SPLIT MANIFEST --
verified to cover all 45 sessions (indices 0-44, contiguous), not just the
~15 training sessions, so a held-out session's index is always valid. A
held-out session's day-layer slice was never touched by training and
remains at exact identity (`docs/DESIGN.md` section 10; verified
empirically against a real trained checkpoint: max|output-input| =
0.000e+00, bias exactly zero for an untrained session's slice) -- so
evaluating a held-out session here is exactly the day-0, no-calibration
decoder (`docs/DESIGN.md` section 5, Arm 0), not an approximation of it.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from b2t_drift.config import load_yaml
from b2t_drift.data.collate import collate_trials
from b2t_drift.data.dataset import TrialDataset
from b2t_drift.data.splits import load_split, split_to_trial_ids
from b2t_drift.evaluation.edit_distance import EditCounts
from b2t_drift.evaluation.metrics import per_session_metrics
from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.phonemes import BLANK_INDEX
from b2t_drift.training.checkpoint import load_checkpoint
from b2t_drift.training.loop import evaluate
from b2t_drift.utils.provenance import get_git_commit_hash

DEFAULT_DATA_CONFIG_PATH = Path("configs/data/t15.yaml")
BATCH_SIZE = 32
NUM_WORKERS = 2

CSV_FIELDNAMES = [
    "session_id", "session_date", "days_since_cut", "n_trials", "passes_threshold",
    "per", "per_ci_lower", "per_ci_upper",
    "substitutions", "deletions", "insertions", "n_reference_phonemes",
    "substitution_rate", "deletion_rate", "insertion_rate",
    "silence_deletions", "phoneme_deletions",
    "median_seq_len", "decoded_length_ratio", "eval_seconds",
]


def _build_model(model_config: dict, n_sessions: int) -> GRUCTC:
    return GRUCTC(
        n_sessions=n_sessions,
        hidden_size=model_config["hidden_size"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        bidirectional=model_config["bidirectional"],
        use_day_layers=model_config["use_day_layers"],
        input_dropout=model_config["input_dropout"],
        smoothing_sigma_bins=model_config["smoothing_sigma_bins"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a checkpoint file (e.g. runs/<run>/checkpoint_best.pt).",
    )
    parser.add_argument(
        "--split", type=Path, required=True,
        help="Path to a split manifest JSON (e.g. splits/temporal_s01_s15_switchboard.json).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory; defaults to the checkpoint's parent directory (its run dir).",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=("cuda", "cpu"),
        help="Device to evaluate on; defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--passing-only", action="store_true",
        help=(
            "Evaluate only eval sessions with passes_threshold true. Default: evaluate ALL "
            "held-out sessions and record passes_threshold per row, so the threshold can be "
            "applied at plotting time rather than baked in here."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out_dir = args.out if args.out is not None else args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_yaml(DEFAULT_DATA_CONFIG_PATH)
    data_root = Path(data_config["raw_data_dir"])

    split = load_split(args.split)
    session_to_index = split["session_to_index"]

    # Peek at the checkpoint's saved config/metadata (weights_only=True is
    # safe here -- this is a checkpoint this project's own code wrote, not
    # the externally-sourced pretrained one that needed the numpy fallback
    # in training/checkpoint.py) before constructing a model to load into.
    peek = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config = peek["config"]["model"]
    checkpoint_epoch = peek["epoch"]
    checkpoint_val_per = peek["metrics"].get("val_per")
    del peek

    model = _build_model(model_config, len(session_to_index)).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.to(device)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    eval_sessions = split["eval_sessions"]
    if args.passing_only:
        eval_sessions = [e for e in eval_sessions if e["passes_threshold"]]

    combined_counts_by_session: dict[str, list[EditCounts]] = {}
    session_metadata: dict[str, dict] = {}
    per_session_extra: dict[str, dict] = {}
    skipped_zero_trials: list[str] = []

    for entry in eval_sessions:
        session_id = entry["session_id"]
        if entry["n_trials"] == 0:
            skipped_zero_trials.append(session_id)
            continue

        trial_ids = split_to_trial_ids(split, data_root, session_id)
        dataset = TrialDataset(data_root, trial_ids, session_to_index)
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, collate_fn=collate_trials, num_workers=NUM_WORKERS,
        )

        result = evaluate(model, dataloader, criterion, device, blank_index=BLANK_INDEX, compute_per=True)
        counts = result["counts_by_session"].get(session_id, [])

        combined_counts_by_session[session_id] = counts
        session_metadata[session_id] = {
            "session_date": entry["session_date"],
            "days_since_cut": entry["days_since_cut"],
        }

        substitutions = sum(c.substitutions for c in counts)
        deletions = sum(c.deletions for c in counts)
        insertions = sum(c.insertions for c in counts)
        n_ref = sum(c.reference_length for c in counts)
        per_session_extra[session_id] = {
            "passes_threshold": entry["passes_threshold"],
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "substitution_rate": substitutions / n_ref if n_ref else 0.0,
            "deletion_rate": deletions / n_ref if n_ref else 0.0,
            "insertion_rate": insertions / n_ref if n_ref else 0.0,
            "silence_deletions": result["silence_deletions"],
            "phoneme_deletions": result["phoneme_deletions"],
            "decoded_length_ratio": result["decoded_to_reference_length_ratio"],
            "eval_seconds": result["seconds"],
        }
        print(
            f"[03_evaluate] {session_id} (days_since_cut={entry['days_since_cut']}): "
            f"n_trials={result['n_trials']}  per={result['per']:.4f}  "
            f"seconds={result['seconds']:.2f}"
        )

    if skipped_zero_trials:
        print(
            f"[03_evaluate] skipped {len(skipped_zero_trials)} eval session(s) with 0 trials "
            f"after corpus filtering (PER undefined for zero trials): {skipped_zero_trials}"
        )

    metrics_rows = per_session_metrics(combined_counts_by_session, session_metadata)

    out_rows = []
    for row in metrics_rows:
        extra = per_session_extra[row["session_id"]]
        out_rows.append({
            "session_id": row["session_id"],
            "session_date": row["session_date"],
            "days_since_cut": row["days_since_cut"],
            "n_trials": row["n_trials"],
            "passes_threshold": extra["passes_threshold"],
            "per": row["per"],
            "per_ci_lower": row["per_ci_lower"],
            "per_ci_upper": row["per_ci_upper"],
            "substitutions": extra["substitutions"],
            "deletions": extra["deletions"],
            "insertions": extra["insertions"],
            "n_reference_phonemes": row["n_reference_phonemes"],
            "substitution_rate": extra["substitution_rate"],
            "deletion_rate": extra["deletion_rate"],
            "insertion_rate": extra["insertion_rate"],
            "silence_deletions": extra["silence_deletions"],
            "phoneme_deletions": extra["phoneme_deletions"],
            "median_seq_len": row["median_seq_len"],
            "decoded_length_ratio": extra["decoded_length_ratio"],
            "eval_seconds": extra["eval_seconds"],
        })

    csv_path = out_dir / "per_session_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[03_evaluate] wrote {len(out_rows)} row(s) to {csv_path}")

    summary = {
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_val_per": checkpoint_val_per,
        "split_name": split["name"],
        "split_path": str(args.split),
        "cut_date": split["cut_date"],
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.now(UTC).isoformat(),
        "device": str(device),
        "passing_only": args.passing_only,
        "n_sessions_evaluated": len(out_rows),
        "n_sessions_skipped_zero_trials": len(skipped_zero_trials),
    }
    summary_path = out_dir / "evaluation_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[03_evaluate] wrote {summary_path}")


if __name__ == "__main__":
    main()
