"""Tests for `scripts/02_train.py`'s CLI-argument resolution.

Only `_apply_epoch_overrides` is unit-tested here -- `main()` itself needs
real data, a model, and a full training loop to exercise, which is out of
scope for a fast unit test (`scripts/03_evaluate.py`/`04_drift_curve.py`
have no dedicated test files either, for the same reason). `scripts/` is a
thin CLI wrapper (`CLAUDE.md`), so the only script-level logic worth a unit
test is the `--max-epochs`/`--patience` override precedence itself.

Loaded via `importlib` (not `import scripts.02_train`) because `02_train`
is not a valid Python module name -- `scripts/` has no `__init__.py` and is
never installed as a package.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02_train.py"


def _load_02_train():
    spec = importlib.util.spec_from_file_location("_test_02_train", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_max_epochs_and_patience_overrides_land_in_resolved_training_config() -> None:
    train_script = _load_02_train()
    args = argparse.Namespace(max_epochs=100, patience=1000)
    training_config = {"max_epochs": 50, "patience": 5, "batch_size": 32}

    max_epochs, patience, effective = train_script._apply_epoch_overrides(
        training_config, max_epochs=50, patience=5, args=args,
    )

    assert max_epochs == 100
    assert patience == 1000
    assert effective["max_epochs"] == 100
    assert effective["patience"] == 1000
    assert effective["batch_size"] == 32  # untouched keys still pass through
    assert training_config == {"max_epochs": 50, "patience": 5, "batch_size": 32}  # not mutated


def test_no_cli_override_leaves_values_unchanged() -> None:
    train_script = _load_02_train()
    args = argparse.Namespace(max_epochs=None, patience=None)
    training_config = {"max_epochs": 50, "patience": 5}

    max_epochs, patience, effective = train_script._apply_epoch_overrides(
        training_config, max_epochs=50, patience=5, args=args,
    )

    assert (max_epochs, patience) == (50, 5)
    assert effective == {"max_epochs": 50, "patience": 5}


def test_max_epochs_override_applies_on_top_of_smoke_adjusted_value() -> None:
    """--smoke adjusts max_epochs/patience before _apply_epoch_overrides
    ever runs (scripts/02_train.py's main()) -- an explicit --max-epochs
    must still win over that adjustment, not just over the raw config."""
    train_script = _load_02_train()
    args = argparse.Namespace(max_epochs=100, patience=None)
    training_config = {"max_epochs": 50, "patience": 5}

    # Simulates main()'s --smoke branch: max_epochs=2 (SMOKE_MAX_EPOCHS),
    # patience=max(5, 2)=5 -- neither equal to the raw config's max_epochs.
    max_epochs, patience, effective = train_script._apply_epoch_overrides(
        training_config, max_epochs=2, patience=5, args=args,
    )

    assert max_epochs == 100
    assert patience == 5
    assert effective["max_epochs"] == 100
