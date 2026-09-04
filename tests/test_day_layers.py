"""Unit tests for `b2t_drift.models.day_layers.DayLayers`.

Targets the fixes from the "Six tasks" review: the transform-convention
docstring fix (Task 2a, verified by a direct interpretation check, not
just a shape check), the three `parameterization` options (Task 2b), the
`_active_session` state_dict round-trip (Task 2c), and the gradient hook
surviving `.to(device)` (Task 2d).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b2t_drift.models.day_layers import DayLayers

PARAMETERIZATIONS = ("full", "diagonal", "lowrank")


# --- Task 2(a): transform convention / interpretation -----------------------


def test_forward_matches_x_at_W_plus_b_convention() -> None:
    """Directly verifies y_t = x_t @ W[s] + b[s] (not the transpose), by
    constructing a non-identity, ASYMMETRIC W and checking the matmul
    order the module actually implements against a hand-computed
    reference. Asymmetric W is essential here -- a symmetric W would not
    distinguish x@W from W@x=  (x@W^T), silently passing either way."""
    day_layers = DayLayers(n_sessions=1, n_features=3)
    with torch.no_grad():
        day_layers.weights[0] = torch.tensor([
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        day_layers.biases[0] = torch.tensor([0.0, 0.0, 5.0])

    x = torch.tensor([[[1.0, 0.0, 0.0]]])  # (batch=1, time=1, features=3)
    session_indices = torch.zeros(1, dtype=torch.int64)
    out = day_layers(x, session_indices)

    # y = x @ W + b: row [1,0,0] @ W = [1,2,0] (row 0 of W), + b = [1,2,5]
    expected = torch.tensor([[[1.0, 2.0, 5.0]]])
    assert torch.allclose(out, expected)


# --- Task 2(b): parameterization option -------------------------------------


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_identity_init_for_each_parameterization(parameterization) -> None:
    day_layers = DayLayers(n_sessions=3, n_features=8, parameterization=parameterization, rank=4)
    x = torch.randn(5, 10, 8)
    session_indices = torch.tensor([0, 1, 2, 0, 1])
    out = day_layers(x, session_indices)
    assert torch.allclose(out, x, atol=1e-6)


def test_full_parameter_count_matches_arithmetic() -> None:
    n_sessions, n_features = 45, 512
    day_layers = DayLayers(n_sessions=n_sessions, n_features=n_features, parameterization="full")
    total = sum(p.numel() for p in day_layers.parameters())
    expected_per_session = n_features * n_features + n_features  # W + bias
    assert expected_per_session == 262_656
    assert total == n_sessions * expected_per_session


def test_diagonal_parameter_count_matches_arithmetic() -> None:
    n_sessions, n_features = 45, 512
    day_layers = DayLayers(n_sessions=n_sessions, n_features=n_features, parameterization="diagonal")
    total = sum(p.numel() for p in day_layers.parameters())
    expected_per_session = n_features + n_features  # scale + bias
    assert expected_per_session == 1_024
    assert total == n_sessions * expected_per_session


def test_lowrank_parameter_count_matches_arithmetic() -> None:
    n_sessions, n_features, rank = 45, 512, 16
    day_layers = DayLayers(n_sessions=n_sessions, n_features=n_features, parameterization="lowrank", rank=rank)
    total = sum(p.numel() for p in day_layers.parameters())
    expected_per_session = n_features * rank * 2 + n_features  # U + V + bias
    assert expected_per_session == 16_896
    assert total == n_sessions * expected_per_session


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_output_shape_for_each_parameterization(parameterization) -> None:
    day_layers = DayLayers(n_sessions=4, n_features=16, parameterization=parameterization, rank=4)
    x = torch.randn(3, 7, 16)
    session_indices = torch.tensor([0, 3, 1])
    out = day_layers(x, session_indices)
    assert out.shape == (3, 7, 16)


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_freeze_all_except_works_for_each_parameterization(parameterization) -> None:
    """Checks the actual masking contract -- every NON-active session's
    slice is exactly zero, for every per-session parameter -- separately
    from whether the active session's true (unmasked) gradient happens
    to be nonzero for a given parameter. The two are not the same thing:
    under "lowrank", V's gradient is legitimately zero at this
    initialization regardless of freezing (dL/dV depends on U, and U
    starts at zero -- see day_layers.py's module docstring on
    symmetry-breaking), so asserting "active session must be nonzero"
    per-parameter would be a wrong expectation for V specifically, not a
    freezing bug. The `any_active_nonzero` check below instead confirms
    SOMETHING is flowing to the active session in aggregate, which is
    enough to know the hook isn't masking everything.
    """
    day_layers = DayLayers(n_sessions=3, n_features=4, parameterization=parameterization, rank=2)
    day_layers.freeze_all_except(1)

    x = torch.randn(3, 5, 4)
    session_indices = torch.tensor([0, 1, 2])
    out = day_layers(x, session_indices)
    out.sum().backward()

    any_active_nonzero = False
    for param in day_layers._session_params:
        assert param.grad is not None
        for s in range(3):
            if s == 1:
                if param.grad[s].abs().sum() > 0:
                    any_active_nonzero = True
            else:
                assert torch.all(param.grad[s] == 0)
    assert any_active_nonzero


def test_invalid_parameterization_raises() -> None:
    with pytest.raises(ValueError, match="parameterization"):
        DayLayers(n_sessions=1, parameterization="bogus")


def test_diagonal_cannot_represent_channel_swap_but_full_can(capsys) -> None:
    """Fits each parameterization (single session) to a known channel-swap
    target and compares final loss -- the concrete demonstration behind
    the module docstring's attribution claim: 'diagonal' cannot mix
    channels by construction, 'full' can represent the swap exactly."""
    torch.manual_seed(0)
    n_features = 8
    x = torch.randn(256, 1, n_features)
    target = x.clone()
    target[:, :, [0, 1]] = target[:, :, [1, 0]]  # swap channels 0 and 1
    session_indices = torch.zeros(256, dtype=torch.int64)

    def fit(parameterization: str, steps: int = 400, lr: float = 0.05) -> float:
        torch.manual_seed(0)
        day_layers = DayLayers(n_sessions=1, n_features=n_features, parameterization=parameterization)
        optimizer = torch.optim.Adam(day_layers.parameters(), lr=lr)
        loss = torch.tensor(float("nan"))
        for _ in range(steps):
            optimizer.zero_grad()
            out = day_layers(x, session_indices)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
        return loss.item()

    full_loss = fit("full")
    diagonal_loss = fit("diagonal")

    print(f"\n[Task 2(b)] channel-swap fit -- final MSE: full={full_loss:.6f}, "
          f"diagonal={diagonal_loss:.6f}")

    assert full_loss < 1e-3, "'full' should fit an exact channel permutation almost perfectly"
    assert diagonal_loss > 0.1, "'diagonal' cannot mix channels, so it should NOT fit the swap"
    assert diagonal_loss > full_loss * 50


# --- Task 2(c): _active_session persists in state_dict ----------------------


def test_active_session_round_trips_through_state_dict() -> None:
    day_layers = DayLayers(n_sessions=4, n_features=6)
    day_layers.freeze_all_except(2)

    reloaded = DayLayers(n_sessions=4, n_features=6)
    reloaded.load_state_dict(day_layers.state_dict())

    assert reloaded._active_session == 2


def test_active_session_none_round_trips_too() -> None:
    day_layers = DayLayers(n_sessions=4, n_features=6)
    reloaded = DayLayers(n_sessions=4, n_features=6)
    reloaded.load_state_dict(day_layers.state_dict())
    assert reloaded._active_session is None


# --- Task 2(d): gradient hook survives .to(device) --------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gradient_hook_survives_to_device() -> None:
    day_layers = DayLayers(n_sessions=3, n_features=4).to("cuda")
    day_layers.freeze_all_except(1)

    x = torch.randn(3, 5, 4, device="cuda")
    session_indices = torch.tensor([0, 1, 2], device="cuda")
    out = day_layers(x, session_indices)
    out.sum().backward()

    assert day_layers.weights.grad is not None
    for s in range(3):
        if s == 1:
            assert day_layers.weights.grad[s].abs().sum() > 0
        else:
            assert torch.all(day_layers.weights.grad[s] == 0)
