"""
Tests for PDE base class.
"""

from __future__ import annotations

import pytest
import torch

from solarpinn.nn.module import Module
from solarpinn.pde.equation import PDE

class DummyModel(Module):

    def forward(
            self,
            x: torch.Tensor
            )-> torch.Tensor:
        return x

class DummyPDE(PDE):
    def residual(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:

        return torch.ones_like(inputs)


def test_call()-> None:
    pde = DummyPDE()

    model = DummyModel()

    x = torch.tensor([[1.0]])

    assert torch.allclose(
            pde(model, x),
            torch.ones_like(x)
            )

def test_repr()-> None:

    assert repr(DummyPDE()) == "DummyPDE"

def test_abstract()-> None:

    class BadPDE(PDE):
        pass
    
    with pytest.raises(TypeError):
        BadPDE()


