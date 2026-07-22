"""
Tests for the PDE base class.
"""

from __future__ import annotations

import torch

from solarpinn.nn.module import Module
from solarpinn.pinn.pde import PDE

class IdentityModel(Module):
    """
    Dummy neural network.
    """

    def forward(
            self,
            x: torch.Tensor
            )-> torch.Tensor:
        return x


class DummyPDE(PDE):
    """
    Dummy PDE for testing.
    """

    def residual(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        
        return model(inputs)


def test_residual()-> None:
    model = IdentityModel()

    pde = DummyPDE()

    x = torch.tensor([1.0, 2.0])

    r = pde.residual(model, x)

    assert torch.allclose(r, x)


def test_loss() -> None:

    model = IdentityModel()

    pde = DummyPDE()

    x = torch.tensor([1.0, 2.0])

    expected = torch.mean(x ** 2)

    assert torch.allclose(
        pde.loss(model, x),
        expected,
    )


def test_call() -> None:

    model = IdentityModel()

    pde = DummyPDE()

    x = torch.tensor([3.0])

    assert torch.allclose(
        pde(model, x),
        torch.tensor(9.0),
    )


def test_repr() -> None:

    pde = DummyPDE()

    assert repr(pde) == "DummyPDE"
