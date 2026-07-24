"""
Tests for the Poisson equation.
"""

from __future__ import annotations

import torch

from solarpinn.nn.module import Module
from solarpinn.pde.poisson import PoissonEquation

class SquareModel(Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x**2


class ZeroModel(Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x * 0.0

def test_zero_residual() -> None:

    model = SquareModel()

    pde = PoissonEquation(
        source=lambda x: torch.full_like(x, 2.0),
    )

    x = torch.tensor(
        [3.0],
        requires_grad=True,
    )

    assert torch.allclose(
        pde(model, x),
        torch.zeros_like(x),
    )


def test_nonzero_residual() -> None:

    model = ZeroModel()

    pde = PoissonEquation(
        source=lambda x: torch.ones_like(x),
    )

    x = torch.tensor(
        [2.0],
        requires_grad=True,
    )

    expected = -torch.ones_like(x)

    assert torch.allclose(
        pde(model, x),
        expected,
    )


def test_repr() -> None:

    pde = PoissonEquation(
        source=lambda x: x,
    )

    assert repr(pde) == "PoissonEquation"
