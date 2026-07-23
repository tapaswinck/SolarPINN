"""
Tests for initial conditions.
"""

from __future__ import annotations

import torch

from solarpinn.nn.module import Module

from solarpinn.pinn.initial import (
        InitialCondition,
        ValueInitialCondition,
        DerivativeInitialCondition
        )

class IdentityModel(Module):

    def forward(
            self,
            x: torch.Tensor
            )-> torch.Tensor:
        return x

class ZeroModel(Module):

    def forward(
            self,
            x: torch.Tensor
            )->torch.Tensor:
        return torch.zeros_like(x)

class SquareModel(Module):

    def forward(
            self,
            x: torch.Tensor
            )-> torch.Tensor:
        return x ** 2


class DummyInitialCondition(InitialCondition):
    
    def loss(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:

        return torch.tensor(1.0)



# ==========================================================
# Base InitialCondition
# ==========================================================


def test_call() -> None:

    ic = DummyInitialCondition()

    model = IdentityModel()

    x = torch.tensor([1.0])

    assert ic(model, x) == 1.0


def test_repr() -> None:

    ic = DummyInitialCondition()

    assert repr(ic) == "DummyInitialCondition"


# ==========================================================
# ValueInitialCondition
# ==========================================================


def test_value_initial_zero_loss() -> None:

    model = IdentityModel()

    ic = ValueInitialCondition(
        lambda x: x,
    )

    x = torch.tensor([1.0, 2.0])

    assert torch.allclose(
        ic(model, x),
        torch.tensor(0.0),
    )


def test_value_initial_positive_loss() -> None:

    model = ZeroModel()

    ic = ValueInitialCondition(
        lambda x: x,
    )

    x = torch.tensor([1.0, 2.0])

    expected = torch.mean(x ** 2)

    assert torch.allclose(
        ic(model, x),
        expected,
    )


def test_value_custom_target() -> None:

    model = IdentityModel()

    ic = ValueInitialCondition(
        lambda x: 2 * x,
    )

    x = torch.tensor([1.0, 2.0])

    expected = torch.mean(x ** 2)

    assert torch.allclose(
        ic(model, x),
        expected,
    )


# ==========================================================
# DerivativeInitialCondition
# ==========================================================


def test_derivative_initial_zero_loss() -> None:

    model = SquareModel()

    ic = DerivativeInitialCondition(
        lambda x: 2 * x,
    )

    x = torch.tensor(
        [2.0],
        requires_grad=True,
    )

    assert torch.allclose(
        ic(model, x),
        torch.tensor(0.0),
    )


def test_derivative_initial_positive_loss() -> None:

    model = SquareModel()

    ic = DerivativeInitialCondition(
        lambda x: torch.zeros_like(x),
    )

    x = torch.tensor(
        [2.0],
        requires_grad=True,
    )

    assert ic(model, x) > 0


def test_derivative_constant_model() -> None:

    class ConstantModel(Module):

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:

            return x * 0.0

    model = ConstantModel()

    ic = DerivativeInitialCondition(
        lambda x: torch.zeros_like(x),
    )

    x = torch.tensor(
        [5.0],
        requires_grad=True,
    )

    assert torch.allclose(
        ic(model, x),
        torch.tensor(0.0),
    )
