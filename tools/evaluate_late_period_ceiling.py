"""Evaluate the late-period ceiling checkpoint on its own val partition,
pooled across all 6 late sessions, with a bootstrap 95% CI.

docs/FINDINGS.md: "Late-period ceiling measured directly: the signal
itself, not the recalibration method, limits the +597d result"
(2026-11-04), including its addenda on what the bootstrap CI resamples
(whole trials, pooled edit-distance ratio -- `evaluation.metrics.
bootstrap_error_rate_ci`) and its two caveats (the 93-day pool span
makes 0.6901 a conservative ceiling; it is architecture/data-quantity
-specific, not an information-theoretic bound).

Reproduces PER=0.6901, 95% CI [0.6757, 0.7040] (n=52 post-sentence-dedup
val trials) from committed inputs: `splits/late_period_ceiling_switchboard.
json` (committed) and a checkpoint trained from it via
`tools/build_late_period_ceiling_split.py` + `uv run scripts/02_train.py
--config configs/experiment/switchboard_baseline.yaml --split
splits/late_period_ceiling_switchboard.json --run-name
late_period_ceiling` -- the checkpoint itself is gitignored (`runs/`),
so a clean checkout must re-run that training command first (~11 min on
the measured hardware, REQUIRES A CUDA DEVICE; this script also requires
CUDA for the forward pass).

Run with: uv run tools/evaluate_late_period_ceiling.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from b2t_drift.config import load_yaml
from b2t_drift.data.collate import collate_trials
from b2t_drift.data.dataset import TrialDataset
from b2t_drift.data.splits import load_split, split_to_trial_ids
from b2t_drift.evaluation.metrics import bootstrap_error_rate_ci
from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.phonemes import BLANK_INDEX
from b2t_drift.training.checkpoint import load_checkpoint
from b2t_drift.training.loop import evaluate

DEFAULT_CHECKPOINT = Path("runs/late_period_ceiling/checkpoint_best.pt")
DEFAULT_SPLIT_PATH = Path("splits/late_period_ceiling_switchboard.json")
DEFAULT_DATA_CONFIG = Path("configs/data/t15.yaml")
BATCH_SIZE = 32
NUM_WORKERS = 2
DEFAULT_N_RESAMPLES = 10000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Path to the trained checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--split", type=Path, default=DEFAULT_SPLIT_PATH,
        help="Path to the split manifest used to train it (default: %(default)s).",
    )
    parser.add_argument(
        "--data-config", type=Path, default=DEFAULT_DATA_CONFIG,
        help="Path to the data config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=DEFAULT_N_RESAMPLES,
        help="Bootstrap resample count (default: %(default)s).",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="Bootstrap CI confidence level (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Bootstrap resampling seed (default: %(default)s).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = load_yaml(args.data_config)
    data_root = Path(data_config["raw_data_dir"])

    split = load_split(args.split)
    session_to_index = split["session_to_index"]

    peek = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config = peek["config"]["model"]
    checkpoint_epoch = peek["epoch"]
    checkpoint_val_per = peek["metrics"].get("val_per")
    del peek

    model = GRUCTC(
        n_sessions=len(session_to_index),
        hidden_size=model_config["hidden_size"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        bidirectional=model_config["bidirectional"],
        use_day_layers=model_config["use_day_layers"],
        input_dropout=model_config["input_dropout"],
        smoothing_sigma_bins=model_config["smoothing_sigma_bins"],
        patch_size=model_config.get("patch_size", 1),
        patch_stride=model_config.get("patch_stride", 1),
        day_layer_parameterization=model_config.get("day_layer_parameterization", "full"),
        day_layer_rank=model_config.get("day_layer_rank", 16),
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.to(device)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    val_trial_ids = split_to_trial_ids(split, data_root, "val")
    dataset = TrialDataset(data_root, val_trial_ids, session_to_index)
    val_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_trials, num_workers=NUM_WORKERS,
    )

    result = evaluate(model, val_loader, criterion, device, blank_index=BLANK_INDEX, compute_per=True)
    pooled_counts = [c for counts in result["counts_by_session"].values() for c in counts]

    n_val = len(pooled_counts)
    point, ci_lower, ci_upper = bootstrap_error_rate_ci(
        pooled_counts, n_resamples=args.n_resamples, confidence=args.confidence, seed=args.seed,
    )

    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint_epoch}, logged val_per={checkpoint_val_per})")
    print(f"val trials evaluated (post-dedup): {n_val}  (manifest val count: {len(val_trial_ids)})")
    print(f"pooled PER (direct from evaluate()): {result['per']:.4f}")
    print(f"bootstrap PER point estimate: {point:.4f}")
    print(f"bootstrap {args.confidence:.0%} CI: [{ci_lower:.4f}, {ci_upper:.4f}]")


if __name__ == "__main__":
    main()
