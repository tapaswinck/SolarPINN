"""
Initial conditoin implementation for PINNs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from solarpinn.nn.module import Module
from solarpinn.physics.gradients import gradient

class InitialCondition(ABC):
    """
    Base class for all initial conditions.
    """

    @abstractmethod
    def loss(
            self,
            model: Module,
            *args,
            **kwargs
            )-> torch.Tensor:
        """
        Compute the initial condition loss.
        """

        raise NotImplementedError

    def __call__(
            self,
            model:Module,
            *args,
            **kwargs
            )-> torch.Tensor:
        """
        Evaluate the initial condition.
        """

        return self.loss(
                model,
                *args,
                **kwargs
                )

    def __repr__(self)-> str:
        return self.__class__.__name__


class ValueInitialCondition(InitialCondition):
    """
    Initial value condition.

    Enforces
        u(x, 0) = f(x)
    """

    def __init__(
            self,
            target: Callable[[torch.Tensor], torch.Tensor]
            )->None:
        self.target = target


    def loss(
            self,
            model:Module,
            inputs: torch.Tensor
            )-> torch.Tensor:

        prediction = model(inputs)

        target = self.target(inputs)

        return torch.mean((prediction - target ) ** 2)


class DerivativeInitialCondition(InitialCondition):
    """
    Initial derivative conditon.

    Enforces
        du/dt(x, 0) = g(x)
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

        grad = gradient(prediction, inputs)

        target = self.target(inputs)


        return torch.mean((grad - target) ** 2)


