"""
Unit tests for loss functions.
"""

from __future__ import annotations

import pytest
import torch

from solarpinn.nn.losses import (
    MSELoss,
    L1Loss,
    SmoothL1Loss,
    HuberLoss,
    RMSELoss,
    RelativeL2Loss,
)


def test_mse_loss() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 3.0, 2.0])

    loss = MSELoss()

    expected = torch.tensor(2.0 / 3.0)

    assert torch.allclose(loss(prediction, target), expected)


def test_l1_loss() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 3.0, 2.0])

    loss = L1Loss()

    expected = torch.tensor(2.0 / 3.0)

    assert torch.allclose(loss(prediction, target), expected)


def test_smooth_l1_loss() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 3.0, 2.0])

    loss = SmoothL1Loss()

    value = loss(prediction, target)

    assert value >= 0
    assert torch.isfinite(value)


def test_huber_loss() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 3.0, 2.0])

    loss = HuberLoss()

    value = loss(prediction, target)

    assert value >= 0
    assert torch.isfinite(value)


def test_rmse_loss() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 3.0, 2.0])

    loss = RMSELoss()

    expected = torch.sqrt(torch.tensor(2.0 / 3.0))

    assert torch.allclose(loss(prediction, target), expected)


def test_relative_l2_loss() -> None:
    prediction = torch.tensor([2.0, 2.0])
    target = torch.tensor([1.0, 1.0])

    loss = RelativeL2Loss()

    expected = torch.tensor(1.0)

    assert torch.allclose(loss(prediction, target), expected)


def test_zero_loss() -> None:
    x = torch.randn(10)

    assert MSELoss()(x, x) == 0
    assert L1Loss()(x, x) == 0
    assert SmoothL1Loss()(x, x) == 0
    assert HuberLoss()(x, x) == 0
    assert RMSELoss()(x, x) < 1e-6
    assert RelativeL2Loss()(x, x) == 0


def test_multidimensional_input() -> None:
    prediction = torch.randn(4, 5)
    target = torch.randn(4, 5)

    losses = [
        MSELoss(),
        L1Loss(),
        SmoothL1Loss(),
        HuberLoss(),
        RMSELoss(),
        RelativeL2Loss(),
    ]

    for loss in losses:
        value = loss(prediction, target)

        assert value.ndim == 0
        assert torch.isfinite(value)


def test_shape_mismatch() -> None:
    prediction = torch.randn(5)
    target = torch.randn(4)

    losses = [
        MSELoss(),
        L1Loss(),
        SmoothL1Loss(),
        HuberLoss(),
        RMSELoss(),
        RelativeL2Loss(),
    ]

    for loss in losses:
        with pytest.raises(AssertionError):
            loss(prediction, target)


def test_constructor_validation() -> None:
    with pytest.raises(AssertionError):
        SmoothL1Loss(beta=0.0)

    with pytest.raises(AssertionError):
        HuberLoss(delta=0.0)