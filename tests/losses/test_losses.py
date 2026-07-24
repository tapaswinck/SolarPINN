"""
Tests for the base Loss class.
"""

from __future__ import annotations

from abc import ABC

import pytest
import torch

from solarpinn.losses.loss import Loss

class DummyLoss(Loss):
    """
    Simple concrete loss for testing.
    """

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
            )-> torch.Tensor:
        return torch.mean((prediction - target) ** 2)


def test_call()-> None:

    loss = DummyLoss()

    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 1.0,1.0])

    expected = torch.mean((prediction - target) ** 2)

    assert torch.allclose(
            loss(prediction, target),
            expected)

def test_repr() -> None:
    loss = DummyLoss()

    assert repr(loss) == "DummyLoss"

def test_abstract_class()-> None:

    class IncompleteLoss(Loss):
        pass

    with pytest.raises(TypeError):
        IncompleteLoss()


