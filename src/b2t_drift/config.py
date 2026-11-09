"""Config loading and validation.

Responsible for reading the YAML files under `configs/data/` and turning
them into a typed `DataConfig` that the rest of the library consumes. No
training or data logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from b2t_drift.phonemes import N_CLASSES


@dataclass
class DataConfig:
    """Typed view of a `configs/data/*.yaml` file.

    Fields match `configs/data/t15.yaml`; see `docs/DATASET.md`
    for how these values were established.
    """

    dataset_name: str
    raw_data_dir: Path
    manifest_path: Path
    n_features: int
    bin_size_ms: int
    n_phoneme_classes: int
    ctc_output_size: int
    blank_index: int
    pad_value: int
    seq_class_ids_padded_length: int


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

    Validates that `ctc_output_size` matches the number of classes in the
    canonical phoneme vocabulary (`b2t_drift.phonemes.N_CLASSES`), since
    the two are defined independently (this file vs. `phonemes.py`) and
    must agree for the CTC output layer to be sized correctly.

    Args:
        path: Path to a `configs/data/*.yaml` file.

    Returns:
        A validated `DataConfig`.

    Raises:
        ValueError: If `ctc_output_size` doesn't match
            `b2t_drift.phonemes.N_CLASSES`.
    """
    raw = load_yaml(path)
    config = DataConfig(
        dataset_name=raw["dataset_name"],
        raw_data_dir=Path(raw["raw_data_dir"]),
        manifest_path=Path(raw["manifest_path"]),
        n_features=raw["n_features"],
        bin_size_ms=raw["bin_size_ms"],
        n_phoneme_classes=raw["n_phoneme_classes"],
        ctc_output_size=raw["ctc_output_size"],
        blank_index=raw["blank_index"],
        pad_value=raw["pad_value"],
        seq_class_ids_padded_length=raw["seq_class_ids_padded_length"],
    )
    if config.ctc_output_size != N_CLASSES:
        raise ValueError(
            f"{path}: ctc_output_size={config.ctc_output_size} does not "
            f"match b2t_drift.phonemes.N_CLASSES={N_CLASSES}"
        )
    return config
