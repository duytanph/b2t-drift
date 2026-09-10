"""Regression test for `scripts/03_evaluate.py::_build_model`.

`_build_model` reconstructs a `GRUCTC` from a checkpoint's saved model
config, and `load_checkpoint` then pours the trained weights into it. If
`_build_model` drops a SHAPE-relevant kwarg (`patch_size`,
`day_layer_parameterization`, ...) the strict `load_state_dict` raises --
noisy, but caught. If it drops a BEHAVIOR-only kwarg (`patch_stride`) the
load SUCCEEDS and every PER the script reports is silently wrong, because
the eval model strides the time axis differently than the model the
checkpoint was actually trained with. That is the bug this pins: an
evaluation script quietly constructing a different architecture than the
checkpoint, with no error raised (`CLAUDE.md`'s "known pitfalls").

`scripts/` has no `__init__.py` and `03_evaluate` is not a valid module
name, so the script is loaded via `importlib` -- the same pattern as
`tests/test_02_train.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from b2t_drift.models.gru_ctc import GRUCTC
from b2t_drift.training.checkpoint import load_checkpoint, save_checkpoint

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_evaluate.py"
N_SESSIONS = 6

# The architecture settled by the round-5 comparison
# (configs/experiment/switchboard_baseline.yaml): patched input + diagonal
# day layers. Reduced hidden_size/n_layers only to keep the test fast --
# the kwargs that matter for reconstruction (patch_size, patch_stride,
# day_layer_parameterization) are the real ones.
CURRENT_ARCH_MODEL_CONFIG = {
    "hidden_size": 64,
    "n_layers": 2,
    "dropout": 0.4,
    "bidirectional": False,
    "use_day_layers": True,
    "day_layer_parameterization": "diagonal",
    "input_dropout": 0.0,
    "smoothing_sigma_bins": 2.0,
    "patch_size": 14,
    "patch_stride": 4,
}


def _load_03_evaluate():
    spec = importlib.util.spec_from_file_location("_test_03_evaluate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _explicit_model(model_config: dict) -> GRUCTC:
    """A GRUCTC built by naming every kwarg explicitly -- the reference
    `_build_model` must reproduce."""
    return GRUCTC(
        n_sessions=N_SESSIONS,
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
    )


def _assert_same_architecture(reference: GRUCTC, rebuilt: GRUCTC) -> None:
    ref_sd, new_sd = reference.state_dict(), rebuilt.state_dict()
    assert set(new_sd) == set(ref_sd), (
        "state_dict keys differ -- _build_model built a different architecture:\n"
        f"  only in checkpoint model: {sorted(set(ref_sd) - set(new_sd))}\n"
        f"  only in rebuilt model:    {sorted(set(new_sd) - set(ref_sd))}"
    )
    for key in ref_sd:
        assert new_sd[key].shape == ref_sd[key].shape, (
            f"{key}: checkpoint shape {tuple(ref_sd[key].shape)} "
            f"!= rebuilt {tuple(new_sd[key].shape)}"
        )


def test_build_model_reconstructs_current_architecture_exactly(tmp_path: Path) -> None:
    module = _load_03_evaluate()

    trained = _explicit_model(CURRENT_ARCH_MODEL_CONFIG)
    ckpt = tmp_path / "checkpoint_best.pt"
    save_checkpoint(
        ckpt, trained, torch.optim.Adam(trained.parameters()), epoch=0,
        metrics={"val_per": 0.17},
        config={"model": CURRENT_ARCH_MODEL_CONFIG, "training": {}},
    )

    rebuilt = module._build_model(CURRENT_ARCH_MODEL_CONFIG, N_SESSIONS)

    _assert_same_architecture(trained, rebuilt)

    # Behavior-only kwargs leave no state_dict footprint -- assert them
    # directly, since dropping patch_stride would load cleanly and corrupt
    # every reported PER.
    assert rebuilt.patch_size == 14
    assert rebuilt.patch_stride == 4
    assert rebuilt.day_layer_parameterization == "diagonal"
    assert rebuilt.day_layers is not None
    assert hasattr(rebuilt.day_layers, "scale")  # diagonal, not "full"'s weights

    # The whole point: the real load path succeeds with no error.
    load_checkpoint(ckpt, rebuilt)


def test_build_model_reproduces_pre_patching_checkpoint_via_get_defaults(tmp_path: Path) -> None:
    """A checkpoint saved before patch_size / day_layer_parameterization
    were config keys: `_build_model`'s `.get()` defaults must rebuild the
    old architecture (patch_size 1, "full" day layers) so old checkpoints
    still load."""
    module = _load_03_evaluate()

    legacy_config = {
        "hidden_size": 32,
        "n_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        "use_day_layers": True,
        "input_dropout": 0.0,
        "smoothing_sigma_bins": 2.0,
    }
    legacy = _explicit_model(legacy_config)
    ckpt = tmp_path / "legacy.pt"
    save_checkpoint(
        ckpt, legacy, torch.optim.Adam(legacy.parameters()), epoch=0,
        metrics={}, config={"model": legacy_config, "training": {}},
    )

    rebuilt = module._build_model(legacy_config, N_SESSIONS)

    _assert_same_architecture(legacy, rebuilt)
    assert rebuilt.patch_size == 1
    assert rebuilt.patch_stride == 1
    assert rebuilt.day_layer_parameterization == "full"
    load_checkpoint(ckpt, rebuilt)
