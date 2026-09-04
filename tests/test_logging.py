"""Unit tests for `b2t_drift.training.logging`.

`wandb` and `mlflow` are NOT installed in this environment (deliberately --
`pyproject.toml`'s optional-dependencies), so their tests exercise the
guarded-import failure path for real, not mocked. `tensorboard` IS
installed (`uv sync --extra tensorboard`), so `TensorBoardLogger` is
tested for real too, including that it produces actual event files.
"""

from __future__ import annotations

import csv
import importlib.util

import pytest

from b2t_drift.training.logging import (
    ConsoleLogger,
    CSVLogger,
    MLflowLogger,
    MultiLogger,
    RunLogger,
    TensorBoardLogger,
    WandbLogger,
)

WANDB_INSTALLED = importlib.util.find_spec("wandb") is not None
MLFLOW_INSTALLED = importlib.util.find_spec("mlflow") is not None
TENSORBOARD_INSTALLED = importlib.util.find_spec("tensorboard") is not None


class _RecordingLogger:
    """A minimal, dependency-free RunLogger for testing MultiLogger fan-out."""

    def __init__(self) -> None:
        self.scalars_calls: list[tuple[int, dict]] = []
        self.config_calls: list[dict] = []
        self.artifact_calls: list = []
        self.finished = False

    def log_scalars(self, step, scalars):
        self.scalars_calls.append((step, scalars))

    def log_config(self, config):
        self.config_calls.append(config)

    def log_artifact(self, path):
        self.artifact_calls.append(path)

    def finish(self):
        self.finished = True


def test_recording_logger_satisfies_runlogger_protocol() -> None:
    assert isinstance(_RecordingLogger(), RunLogger)


# --- ConsoleLogger -----------------------------------------------------


def test_console_logger_prints_scalars(capsys) -> None:
    logger = ConsoleLogger()
    logger.log_scalars(3, {"train_loss": 1.2345, "n_batches": 10})
    out = capsys.readouterr().out
    assert "step 3" in out
    assert "train_loss=1.2345" in out
    assert "n_batches=10" in out


def test_console_logger_prints_config(capsys) -> None:
    ConsoleLogger().log_config({"seed": 0, "model": {"hidden_size": 512}})
    out = capsys.readouterr().out
    assert "hidden_size" in out


# --- CSVLogger -----------------------------------------------------------


def test_csv_logger_writes_header_and_rows(tmp_path) -> None:
    logger = CSVLogger(tmp_path)
    logger.log_scalars(0, {"epoch": 0, "train_loss": 1.0, "val_per": 0.5})
    logger.log_scalars(1, {"epoch": 1, "train_loss": 0.8, "val_per": 0.4})

    with logger.path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert logger.path.name == "metrics.csv"
    assert len(rows) == 2
    assert rows[0]["epoch"] == "0"
    assert rows[1]["train_loss"] == "0.8"


def test_csv_logger_raises_on_inconsistent_keys(tmp_path) -> None:
    logger = CSVLogger(tmp_path)
    logger.log_scalars(0, {"epoch": 0, "train_loss": 1.0})
    with pytest.raises(ValueError, match="columns"):
        logger.log_scalars(1, {"epoch": 1, "val_loss": 0.9})  # different keys


def test_csv_logger_appends_across_instances(tmp_path) -> None:
    CSVLogger(tmp_path).log_scalars(0, {"epoch": 0, "loss": 1.0})
    second = CSVLogger(tmp_path)  # re-opens the same file
    second.log_scalars(1, {"epoch": 1, "loss": 0.5})

    with second.path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


# --- TensorBoardLogger -----------------------------------------------------


@pytest.mark.skipif(not TENSORBOARD_INSTALLED, reason="tensorboard not installed")
def test_tensorboard_logger_writes_event_files(tmp_path) -> None:
    logger = TensorBoardLogger(tmp_path)
    logger.log_scalars(0, {"train_loss": 1.0})
    logger.log_config({"seed": 0})
    logger.finish()

    tb_dir = tmp_path / "tb"
    assert tb_dir.exists()
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    assert len(event_files) >= 1


# --- WandbLogger: guarded import ------------------------------------------


@pytest.mark.skipif(WANDB_INSTALLED, reason="wandb IS installed; this tests the absent case")
def test_wandb_logger_raises_clear_error_when_not_installed() -> None:
    with pytest.raises(ImportError, match="wandb"):
        WandbLogger(project="b2t-drift")


# --- MLflowLogger: guarded import ------------------------------------------


@pytest.mark.skipif(MLFLOW_INSTALLED, reason="mlflow IS installed; this tests the absent case")
def test_mlflow_logger_raises_clear_error_when_not_installed() -> None:
    with pytest.raises(ImportError, match="mlflow"):
        MLflowLogger(experiment_name="b2t-drift")


# --- MultiLogger -----------------------------------------------------------


def test_multi_logger_fans_out_to_every_logger() -> None:
    a, b = _RecordingLogger(), _RecordingLogger()
    multi = MultiLogger([a, b])

    multi.log_scalars(5, {"loss": 1.0})
    multi.log_config({"seed": 1})
    multi.log_artifact("checkpoint.pt")
    multi.finish()

    for logger in (a, b):
        assert logger.scalars_calls == [(5, {"loss": 1.0})]
        assert logger.config_calls == [{"seed": 1}]
        assert logger.artifact_calls == ["checkpoint.pt"]
        assert logger.finished is True


def test_multi_logger_with_console_and_csv_degrades_gracefully(tmp_path, capsys) -> None:
    """Simulates the documented fallback: no optional trackers installed
    or requested, training still logs via Console + CSV."""
    multi = MultiLogger([ConsoleLogger(), CSVLogger(tmp_path)])
    multi.log_scalars(0, {"epoch": 0, "train_loss": 1.0})

    assert "step 0" in capsys.readouterr().out
    assert (tmp_path / "metrics.csv").exists()
