"""CLI: train a model from an experiment config and a split manifest.

Thin wrapper only -- parses arguments, loads the experiment/data configs
and the split manifest, builds the datasets/dataloaders/model/optimizer/
loggers from them, and calls `training.loop.fit`. No logic belongs here
(`CLAUDE.md`): every actual decision (how a split becomes trials, how
trials become batches, what the model looks like, how training proceeds)
is made by library code in `b2t_drift.data` / `b2t_drift.models` /
`b2t_drift.training` that this script only calls.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from b2t_drift.config import load_yaml
from b2t_drift.data.collate import LengthBucketedSampler, collate_trials
from b2t_drift.data.dataset import TrialDataset
from b2t_drift.data.splits import load_split, split_to_trial_ids
from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.phonemes import BLANK_INDEX
from b2t_drift.training.checkpoint import load_checkpoint
from b2t_drift.training.logging import (
    MLflowLogger,
    RunLogger,
    TensorBoardLogger,
    WandbLogger,
)
from b2t_drift.training.loop import fit
from b2t_drift.utils.provenance import capture_provenance, get_git_commit_hash
from b2t_drift.utils.seeding import set_seed

DEFAULT_CONFIG_PATH = Path("configs/experiment/switchboard_baseline.yaml")
SMOKE_MAX_EPOCHS = 2
SMOKE_N_TRAIN = 64
SMOKE_N_VAL = 32


def _build_run_name(split_name: str, given: str | None) -> str:
    """`{date}_{split-name}_{githash}` unless a name was given explicitly."""
    if given:
        return given
    commit = get_git_commit_hash() or "nogit"
    return f"{datetime.now(UTC).date().isoformat()}_{split_name}_{commit[:8]}"


def _build_dataloader(
    data_root: Path,
    split: dict,
    partition: str,
    session_to_index: dict,
    batch_size: int,
    n_workers: int,
    limit: int | None,
    drop_last: bool,
) -> tuple[DataLoader, LengthBucketedSampler]:
    """One `DataLoader` + its `LengthBucketedSampler`, for `partition`
    ("train" or "val") of `split`. `limit` truncates the trial list to
    its first `limit` entries -- used only by `--smoke`."""
    trial_ids = split_to_trial_ids(split, data_root, partition)
    if limit is not None:
        trial_ids = trial_ids[:limit]

    dataset = TrialDataset(data_root, trial_ids, session_to_index)
    sampler = LengthBucketedSampler(trial_ids, batch_size=batch_size, drop_last=drop_last)
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_trials, num_workers=n_workers,
    )
    return dataloader, sampler


def main() -> None:
    """Entry point: load an experiment config + split manifest, run training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to an experiment config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--split", type=Path, required=True,
        help="Path to a split manifest JSON (e.g. splits/temporal_s01_s15_switchboard.json).",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name; defaults to {date}_{split-name}_{githash}.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb-mode", type=str, default="online", choices=("online", "offline"),
        help="wandb mode (default: %(default)s).",
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging.")
    parser.add_argument(
        "--mlflow-uri", type=str, default="file:./mlruns",
        help="MLflow tracking URI (default: %(default)s -- a local, file-based store).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            f"Pipeline verification only: {SMOKE_MAX_EPOCHS} epochs on "
            f"{SMOKE_N_TRAIN} train / {SMOKE_N_VAL} val trials."
        ),
    )
    args = parser.parse_args()

    exp_config = load_yaml(args.config)
    data_config = load_yaml(exp_config["data_config"])
    model_config = exp_config["model"]
    training_config = exp_config["training"]
    logging_config = exp_config.get("logging", {})

    data_root = Path(data_config["raw_data_dir"])
    split = load_split(args.split)
    session_to_index = split["session_to_index"]

    run_name = _build_run_name(split["name"], args.run_name)
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = training_config["seed"]
    set_seed(seed)

    max_epochs = SMOKE_MAX_EPOCHS if args.smoke else training_config["max_epochs"]
    # patience must not be able to cut a deliberately tiny smoke run
    # short before both its epochs run.
    patience = max(training_config["patience"], max_epochs) if args.smoke else training_config["patience"]
    train_limit = SMOKE_N_TRAIN if args.smoke else None
    val_limit = SMOKE_N_VAL if args.smoke else None

    train_loader, train_sampler = _build_dataloader(
        data_root, split, "train", session_to_index,
        batch_size=training_config["batch_size"], n_workers=training_config["num_workers"],
        limit=train_limit, drop_last=True,
    )
    val_loader, _ = _build_dataloader(
        data_root, split, "val", session_to_index,
        batch_size=training_config["batch_size"], n_workers=training_config["num_workers"],
        limit=val_limit, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUCTC(
        n_sessions=len(session_to_index),
        hidden_size=model_config["hidden_size"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        bidirectional=model_config["bidirectional"],
        use_day_layers=model_config["use_day_layers"],
        input_dropout=model_config["input_dropout"],
        smoothing_sigma_bins=model_config["smoothing_sigma_bins"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    use_amp = bool(training_config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    loggers: list[RunLogger] = []
    try:
        loggers.append(TensorBoardLogger(run_dir))
    except ImportError as exc:
        print(f"[02_train] TensorBoard not available, skipping: {exc}")
    if args.wandb:
        loggers.append(WandbLogger(
            project=logging_config.get("wandb_project", "b2t-drift"),
            name=run_name, group=logging_config.get("wandb_group"),
            tags=[split["name"]], mode=args.wandb_mode,
        ))
    if args.mlflow:
        loggers.append(MLflowLogger(
            experiment_name=logging_config.get("mlflow_experiment", "b2t-drift"),
            tracking_uri=args.mlflow_uri, run_name=run_name,
        ))

    resolved_config = {
        "experiment_name": exp_config["experiment_name"],
        "split_name": split["name"],
        "split_path": str(args.split),
        "model": model_config,
        "training": training_config,
        "run_name": run_name,
        "smoke": args.smoke,
    }
    provenance = capture_provenance(seed=seed)
    with (run_dir / "provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2)

    for logger in loggers:
        logger.log_config(resolved_config)

    print(f"[02_train] run_dir={run_dir}  device={device}  n_sessions={len(session_to_index)}  "
          f"train_trials={len(train_sampler.trial_ids)}  smoke={args.smoke}")

    summary = fit(
        model, train_loader, val_loader, optimizer, criterion, device,
        max_epochs=max_epochs, patience=patience, run_dir=run_dir,
        scaler=scaler, grad_clip=training_config["grad_clip"],
        train_sampler=train_sampler, loggers=loggers, config=resolved_config,
    )

    print(f"[02_train] done: {summary}")

    if summary["best_epoch"] is not None:
        # Sanity check: the saved checkpoint_best.pt actually loads back
        # cleanly (training/checkpoint.py's weights_only handling) --
        # cheap, and catches a corrupt-checkpoint failure at train time
        # rather than whenever it's next needed for evaluation.
        sanity_model = GRUCTC(
            n_sessions=len(session_to_index), hidden_size=model_config["hidden_size"],
            n_layers=model_config["n_layers"], dropout=model_config["dropout"],
            bidirectional=model_config["bidirectional"], use_day_layers=model_config["use_day_layers"],
            input_dropout=model_config["input_dropout"],
            smoothing_sigma_bins=model_config["smoothing_sigma_bins"],
        )
        load_checkpoint(run_dir / "checkpoint_best.pt", sanity_model)
        print(f"[02_train] checkpoint_best.pt reloads cleanly (epoch {summary['best_epoch']}).")


if __name__ == "__main__":
    main()
