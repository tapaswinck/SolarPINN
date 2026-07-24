"""
Tests for mean squared error loss.
"""

from __future__ import annotations

import torch

from solarpinn.losses.mse import MSELoss

def test_zero_loss()-> None:
    
    loss = MSELoss()

    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])

    assert torch.allclose(
            loss(prediction, target),
            torch.tensor(0.0)
            )


def test_positive_loss()-> None:
    loss = MSELoss()

    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([0.0, 0.0, 0.0])

    expected = torch.mean((prediction - target) ** 2)

    assert torch.allclose(
            loss(prediction, target),
            expected
            )


def test_matrix_input()-> None:
    loss = MSELoss()

    prediction = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0]
                ]
            )
    target = torch.zeros_like(prediction)

    expected = torch.mean((prediction - target) ** 2)

    assert torch.allclose(
            loss(prediction, target),
            expected
            )

def test_call_matches_forward() -> None:
    loss = MSELoss()

    prediction = torch.randn(5)
    target = torch.randn(5)

    assert torch.allclose(
            loss(prediction, target),
            loss.forward(prediction, target)
            )

def test_repr() -> None:
    
    assert repr(MSELoss()) == "MSELoss"

