"""
Boundary condition implementation for PINNs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from solarpinn.nn.module import Module
from solarpinn.physics.gradients import gradient


class BoundaryCondition(ABC):
    """
    base class for all boundary conditions.
    """

    @abstractmethod
    def loss(
            self,
            model: Module,
            *args,
            **kwargs
            )-> torch.Tensor:
        """
        Compute the boundary loss.
        """
        raise NotImplementedError

    def __call__(
            self,
            model: Module,
            *args,
            **kwargs
            )-> torch.Tensor:
        """
        Evaluate the boundary condition.
        """

        return self.loss(model, *args, **kwargs)


    def __repr__(self)-> str:
        return self.__class__.__name__


class DirichletBoundary(BoundaryCondition):
    """
    Dirichlet boundary conditions.

    Enforces
        u(x) = g(x)
    """

    def __init__(
            self,
            target: Callable[[torch.Tensor], torch.Tensor]
    )-> None:
        self.target = target

    def loss(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        prediction = model(inputs)
        target = self.target(inputs)

        return torch.mean((prediction - target) ** 2)


class NeumannBoundary(BoundaryCondition):
    """
    Neumann boundary condition.

    Enforces
        du/dn = g(x)
    
    where g is the prescribed normal derivative.
    """
    def __init__(
            self,
            target: Callable[[torch.Tensor], torch.Tensor]
            )-> None:
        self.target = target

    def loss(
            self,
            model:Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        prediction = model(inputs)

        grad = gradient(prediction, inputs)

        target = self.target(inputs)

        return torch.mean((grad - target) ** 2)

class RobinBoundary(BoundaryCondition):
    """
    Robin (mixed) boundary condition.

    Enforces 
        a * u + b * du/dn = g(x)
    """

    def __init__(
            self,
            a: float,
            b: float,
            target: Callable[[torch.Tensor], torch.tensor]
            )-> None:
        self.a = a
        self.b = b
        self.target = target


    def loss(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:

        prediction = model(inputs)

        grad = gradient(prediction, inputs)

        lhs = self.a * prediction + self.b * grad
        rhs = self.target(inputs)

        return torch.mean((lhs - rhs) ** 2)

class PeriodicBoundary(BoundaryCondition):
    """
    Periodic boundary condition.

    Enforces
        u(left) = u(right)

    Optionally enforces
        du/dn(left) = du/dn(right)
    """

    def __init__(
            self,
            derivative: bool = False
            )-> None:
        self.derivative = derivative

    def loss(
            self,
            model: Module,
            left: torch.Tensor,
            right: torch.Tensor
            )-> torch.Tensor:
        left_prediction = model(left)
        right_prediction = model(right)

        loss = torch.mean(
                (left_prediction - right_prediction) ** 2)

        if self.derivative:
            left_gradient = gradient(left_prediction, left)
            right_gradient = gradient(right_prediction, right)

            loss = loss + torch.mean((left_gradient - right_gradient) ** 2)

        return loss

