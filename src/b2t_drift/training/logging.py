"""Experiment logging: a `RunLogger` abstraction so training code never
imports `wandb`, `tensorboard`, or `mlflow` directly.

`training/loop.py`'s `fit` accepts a list of `RunLogger`-shaped objects and
fans scalars, config, and artifacts out to all of them via `MultiLogger` --
it never checks which trackers are installed, never branches on tracker
identity, and never imports any tracker package itself. That's entirely
this module's job, isolated here so a missing or broken tracker package
can never break `import b2t_drift.training.loop`.

WHY THREE TRACKERS. The redundancy is deliberate, not accidental --
`docs/DESIGN.md`'s "Model hyperparameters" / cut-point-sweep sections are
the reason a project this size needs more than one:
  - **TensorBoard is the offline multi-run comparison tool.** Every run
    writes event files to `runs/{run_name}/tb/`, so `tensorboard --logdir
    runs/` overlays every run on shared axes with no network, account, or
    server required -- this is the PRIMARY mechanism for comparing the
    four cut-point sweep runs (`docs/DESIGN.md` section 7) in one figure.
  - **wandb is the hosted equivalent**, adding system metrics (GPU
    utilization, memory) and config tracking a local TensorBoard log
    doesn't capture. Pass `group="cutpoint_sweep"` so the four sweep runs
    group together in the wandb UI.
  - **MLflow is the experiment-tracking / model-registry standard** in
    many industry settings. Its strength here is PARAMETER comparison
    across runs and artifact lineage (which config produced which
    checkpoint), rather than curve overlay -- a different query shape
    than TensorBoard's or wandb's.

ALL THREE ARE OPTIONAL, AT BOTH INSTALL AND RUNTIME (`pyproject.toml`'s
`[project.optional-dependencies]`). If none are installed, or none are
requested, the training loop degrades to `ConsoleLogger` + `CSVLogger`
only -- `fit` always includes both of those regardless of what else is
passed (see its docstring), so a run's metrics are never lost to a
missing tracker package. `WandbLogger` and `MLflowLogger` (and, for
symmetry, `TensorBoardLogger`, since `torch.utils.tensorboard.SummaryWriter`
itself requires the separate `tensorboard` package to be installed) guard
their imports inside `__init__` and raise a clear `ImportError` naming the
missing package and how to install it -- never a bare `ModuleNotFoundError`
pointing at some internal import line.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunLogger(Protocol):
    """Structural interface every logger in this module satisfies.

    Not a base class to inherit from -- any object with these four
    methods (matching signatures) is a valid `RunLogger`, which is what
    lets `ConsoleLogger`, `CSVLogger`, and the guarded-import trackers
    below share a `MultiLogger` list without any of them depending on
    each other or on a common ancestor.
    """

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        """Log a batch of scalar metrics at one step (typically an epoch)."""
        ...

    def log_config(self, config: dict[str, Any]) -> None:
        """Log the resolved run configuration, once, near the start of a run."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Log a file (a checkpoint, `provenance.json`, ...) as a run artifact."""
        ...

    def finish(self) -> None:
        """Flush and close this logger. Called once, at the end of a run."""
        ...


class ConsoleLogger:
    """Prints a formatted line per `log_scalars` call. Always available."""

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        formatted = "  ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in scalars.items()
        )
        print(f"[step {step}] {formatted}")

    def log_config(self, config: dict[str, Any]) -> None:
        print(f"[config] {json.dumps(config, indent=2, default=str)}")

    def log_artifact(self, path: Path) -> None:
        print(f"[artifact] {path}")

    def finish(self) -> None:
        pass


class CSVLogger:
    """Appends `log_scalars` rows to `run_dir/metrics.csv`. Always available.

    Every `log_scalars` call after the first must supply the SAME set of
    keys as the first call -- the CSV's column set is fixed by whichever
    call establishes it, and a later call with a different key set raises
    rather than silently writing a misaligned row or growing the header
    (a training loop that calls this with a fixed dict shape every epoch,
    per `training/loop.py`'s `fit`, will never hit this).
    """

    def __init__(self, run_dir: Path, filename: str = "metrics.csv") -> None:
        self.path = Path(run_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="") as f:
                header = next(csv.reader(f), None)
            if header:
                self._fieldnames = header

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(scalars.keys())
            write_header = True
        else:
            if list(scalars.keys()) != self._fieldnames:
                raise ValueError(
                    f"CSVLogger.log_scalars: keys {list(scalars.keys())} do "
                    f"not match this file's established columns "
                    f"{self._fieldnames} ({self.path})."
                )
            write_header = False

        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(scalars)

    def log_config(self, config: dict[str, Any]) -> None:
        pass  # CSV has no natural place for a config blob; see ConsoleLogger.

    def log_artifact(self, path: Path) -> None:
        pass  # CSV does not track files.

    def finish(self) -> None:
        pass


class TensorBoardLogger:
    """Writes to `run_dir/tb/` via `torch.utils.tensorboard.SummaryWriter`.

    Guarded: `SummaryWriter` itself raises if the separate `tensorboard`
    package isn't installed (it is not bundled with `torch`), so this
    class is optional at runtime exactly like `WandbLogger` /
    `MLflowLogger`, even though it goes through `torch.utils`.
    """

    def __init__(self, run_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoardLogger requires the 'tensorboard' package, "
                "which is not installed. Install it with `uv sync --extra "
                "tensorboard` (or `pip install tensorboard`), or omit "
                "TensorBoardLogger from your loggers list."
            ) from exc

        self._writer = SummaryWriter(log_dir=str(Path(run_dir) / "tb"))

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        for key, value in scalars.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, global_step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._writer.add_text("config", f"```\n{json.dumps(config, indent=2, default=str)}\n```")

    def log_artifact(self, path: Path) -> None:
        # TensorBoard has no generic artifact store; record the path so
        # it's at least visible alongside the run's scalars.
        self._writer.add_text("artifact", str(path))

    def finish(self) -> None:
        self._writer.close()


class WandbLogger:
    """Optional: logs to Weights & Biases. Guarded import.

    Supports `mode="offline"` for machines without network access to
    wandb's servers (metrics are written locally and can be `wandb sync`'d
    later) -- the same run script works with or without connectivity by
    changing only this argument.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbLogger requires the 'wandb' package, which is not "
                "installed. Install it with `uv sync --extra wandb` (or "
                "`pip install wandb`), or run without --wandb."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=project, name=name, group=group, tags=tags, mode=mode,
            config=config or {},
        )

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        self._run.log(scalars, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update(config, allow_val_change=True)

    def log_artifact(self, path: Path) -> None:
        artifact = self._wandb.Artifact(Path(path).stem, type="model")
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()


class MLflowLogger:
    """Optional: logs to MLflow. Guarded import.

    Uses a LOCAL, FILE-BASED tracking URI by default (`file:./mlruns` --
    `mlruns/` at wherever the process is run from, conventionally the
    repo root, matching every other relative path in this project). No
    tracking server required; pass `tracking_uri` for a real one.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "file:./mlruns",
        run_name: str | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflowLogger requires the 'mlflow' package, which is not "
                "installed. Install it with `uv sync --extra mlflow` (or "
                "`pip install mlflow`), or run without --mlflow."
            ) from exc

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        numeric = {k: float(v) for k, v in scalars.items() if isinstance(v, (int, float))}
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._mlflow.log_params(_flatten_for_mlflow(config))

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def finish(self) -> None:
        self._mlflow.end_run()


def _flatten_for_mlflow(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config dict into MLflow's flat, string-valued
    `log_params` shape (`{"data.n_features": "512", ...}`)."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_for_mlflow(value, full_key))
        else:
            flat[full_key] = str(value)
    return flat


class MultiLogger:
    """Fans every call out to a list of `RunLogger`s, in order."""

    def __init__(self, loggers: list[RunLogger]) -> None:
        self._loggers = list(loggers)

    def log_scalars(self, step: int, scalars: dict[str, Any]) -> None:
        for logger in self._loggers:
            logger.log_scalars(step, scalars)

    def log_config(self, config: dict[str, Any]) -> None:
        for logger in self._loggers:
            logger.log_config(config)

    def log_artifact(self, path: Path) -> None:
        for logger in self._loggers:
            logger.log_artifact(path)

    def finish(self) -> None:
        for logger in self._loggers:
            logger.finish()
