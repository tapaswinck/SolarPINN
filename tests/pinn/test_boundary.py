"""
Tests for boundary conditions.
"""

from __future__ import annotations

import torch

from solarpinn.nn.module import Module
from solarpinn.pinn.boundary import (
    BoundaryCondition,
    DirichletBoundary,
    NeumannBoundary,
    RobinBoundary,
    PeriodicBoundary
)


class IdentityModel(Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x


class ZeroModel(Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x)


class DummyBoundary(BoundaryCondition):

    def loss(
        self,
        model: Module,
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        return torch.tensor(1.0)

def test_call() -> None:

    bc = DummyBoundary()

    model = IdentityModel()

    x = torch.tensor([1.0])

    assert bc(model, x) == 1.0

def test_repr() -> None:

    bc = DummyBoundary()

    assert repr(bc) == "DummyBoundary"

def test_dirichlet_zero_loss() -> None:

    model = IdentityModel()

    bc = DirichletBoundary(lambda x: x)

    x = torch.tensor([1.0, 2.0])

    assert torch.allclose(
        bc(model, x),
        torch.tensor(0.0),
    )

def test_dirichlet_positive_loss() -> None:

    model = ZeroModel()

    bc = DirichletBoundary(lambda x: x)

    x = torch.tensor([1.0, 2.0])

    expected = torch.mean(x ** 2)

    assert torch.allclose(
        bc(model, x),
        expected,
    )

def test_custom_target() -> None:

    model = IdentityModel()

    bc = DirichletBoundary(lambda x: 2 * x)

    x = torch.tensor([1.0, 2.0])

    expected = torch.mean(x ** 2)

    assert torch.allclose(
        bc(model, x),
        expected,
    )

class SquareModel(Module):

    def forward(
            self,
            x: torch.Tensor
            )-> torch.Tensor:
        return x ** 2


def test_neumann_zero_loss()-> None:
    model = SquareModel()

    bc = NeumannBoundary(lambda x: 2 * x)

    x = torch.tensor([2.0], requires_grad = True)

    assert torch.allclose(bc(model, x), torch.tensor(0.0))


def test_neumann_positive_loss()-> None:
    model = SquareModel()

    bc = NeumannBoundary(lambda x: torch.zeros_like(x))

    x = torch.tensor([2.0], requires_grad = True)

    assert bc(model, x)>0


class ConstantModel(Module):
    def forward(
            self,
            x:torch.Tensor
            )-> torch.Tensor:
        return torch.zeros_like(x)

def test_robin_zero_loss()-> None:

    model = SquareModel()

    bc = RobinBoundary(
            a=1.0,
            b=1.0,
            target = lambda x: x ** 2 + 2 * x
            )

    x = torch.tensor([2.0], requires_grad = True)

    assert torch.allclose(bc(model, x), torch.tensor(0.0))

def test_robin_positive_loss() -> None:

    model = SquareModel()

    bc = RobinBoundary(
            a = 1.0,
            b =1.0,
            target = lambda x: torch.zeros_like(x)
            )

    x = torch.tensor([2.0], requires_grad = True)

    assert bc(model, x) > 0

def test_robin_dirichlet_case()-> None:
    model = SquareModel()

    bc = RobinBoundary(
            a = 1.0,
            b = 0.0,
            target = lambda x: x ** 2
            )

    x = torch.tensor([2.0], requires_grad = True)

    assert torch.allclose(bc(model, x), torch.tensor(0.0))

def test_robin_neumann_case()-> None:
    model = SquareModel()

    bc = RobinBoundary(
            a = 0.0,
            b = 1.0,
            target = lambda x: 2 * x
            )
    x = torch.tensor([2.0], requires_grad = True)

    assert torch.allclose(bc(model, x), torch.tensor(0.0))



def test_periodic_zero_loss()-> None:
    
    model = IdentityModel()

    bc = PeriodicBoundary()

    left = torch.tensor(
            [1.0], requires_grad = True)

    right = torch.tensor([1.0], requires_grad = True)

    assert torch.allclose(bc(model, left, right), torch.tensor(0.0))


def test_periodic_positive_loss()-> None:
    
    model = IdentityModel()
    
    bc = PeriodicBoundary()

    left = torch.tensor([0.0], requires_grad = True)

    right = torch.tensor([1.0], requires_grad = True)

    assert bc(model, left, right) > 0


def test_periodic_derivative_zero_loss()-> None:

    model = SquareModel()

    bc = PeriodicBoundary(derivative = True)

    left = torch.tensor([2.0], requires_grad = True)

    right = torch.tensor([2.0], requires_grad = True)

    assert torch.allclose(bc(model, left, right), torch.tensor(0.0))


def test_periodic_derivative_positive_loss()-> None:
    
    model = SquareModel()

    bc = PeriodicBoundary(derivative = True)

    left = torch.tensor([0.0], requires_grad = True)

    right = torch.tensor([1.0], requires_grad= True)

    assert bc(model, left, right) > 0

def test_periodic_derivative_zero_loss()-> None:
    
    model = SquareModel()

    bc = PeriodicBoundary(derivative = True)

    left = torch.tensor([2.0], requires_grad = True)

    right = torch.tensor([2.], requires_grad = True)

    assert torch.allclose(bc(model, left, right), torch.tensor(0.0))


def test_periodic_repr()-> None:

    bc = PeriodicBoundary()

    assert repr(bc) == "PeriodicBoundary"



