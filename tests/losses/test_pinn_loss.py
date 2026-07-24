"""
Tests for PINNLoss.
"""

from __future__ import annotations

import torch

from solarpinn.losses.pinn_loss import PINNLoss
from solarpinn.nn.module import Module

class ZeroModel(Module):

    def forward(
            self,
            x: torch.Tensor
            )-> torch.Tensor:
        return torch.zeros_like(x)

class ConstantResidual:
    def __init__(self, value: float)-> None:
        self.value = value

    def loss(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        return torch.tensor(self.value)

def test_empty_loss()-> None:
    loss = PINNLoss()

    model = ZeroModel()

    x = torch.tensor([[0.0]])

    assert torch.allclose(
            loss(model, x),
            torch.tensor(0.0)
            )

def test_single_term() -> None:

    loss = PINNLoss()

    loss.add(ConstantResidual(2.0))

    model = ZeroModel()

    x = torch.tensor([[0.0]])

    assert torch.allclose(
            loss(model, x),
            torch.tensor(2.0)
            )

def test_multiple_terms() -> None:
    
    loss = PINNLoss()

    loss.add(ConstantResidual(1.0))
    loss.add(ConstantResidual(2.5))
    loss.add(ConstantResidual(3.5))

    model = ZeroModel()

    x= torch.tensor([[0.0]])

    assert torch.allclose(
            loss(model, x),
            torch.tensor(7.0)
            )

def test_len()-> None:
    loss = PINNLoss()

    assert len(loss) == 0
    
    loss.add(ConstantResidual(1.0))

    assert len(loss) == 1

def test_repr()-> None:
    loss = PINNLoss()

    assert repr(loss) == "PINNLoss(n_terms=0)"

