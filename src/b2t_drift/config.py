"""Config loading and validation.

Responsible for reading the YAML files under `configs/` (data, model, and
experiment configs) and turning them into typed objects that the rest of the
library consumes. No training or data logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Typed view of a `configs/data/*.yaml` file.

    TODO: fields are placeholders until the T15 HDF5 structure is inspected.
    """

    dataset_name: str
    data_root: Path
    num_neural_features: int
    bin_size_ms: int
    num_sessions: int


@dataclass
class ModelConfig:
    """Typed view of a `configs/model/*.yaml` file."""

    model_name: str
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool


@dataclass
class ExperimentConfig:
    """Typed view of a `configs/experiment/*.yaml` file.

    Bundles a `DataConfig` and `ModelConfig` together with run-level
    hyperparameters (seed, epochs, batch size, learning rate).
    """

    experiment_name: str
    data: DataConfig
    model: ModelConfig
    seed: int
    num_epochs: int
    batch_size: int
    learning_rate: float


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a plain dict.

    Args:
        path: Path to a YAML config file.

    Returns:
        The parsed YAML contents as a dict.
    """
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def load_data_config(path: Path) -> DataConfig:
    """Load and validate a data config file into a `DataConfig`.

    Args:
        path: Path to a `configs/data/*.yaml` file.

    Returns:
        A validated `DataConfig`.
    """
    raise NotImplementedError


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate a model config file into a `ModelConfig`.

    Args:
        path: Path to a `configs/model/*.yaml` file.

    Returns:
        A validated `ModelConfig`.
    """
    raise NotImplementedError


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment config, resolving its referenced data/model configs.

    Args:
        path: Path to a `configs/experiment/*.yaml` file.

    Returns:
        A fully resolved `ExperimentConfig`.
    """
    raise NotImplementedError
