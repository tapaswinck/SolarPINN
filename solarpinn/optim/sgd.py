"""
Stochastic Gradient Descent optimizer.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from solarpinn.nn.parameter import Parameter

from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(
            self,
            parameters: Iterable[Parameter],
            lr: float = 1e-3,
            momentum: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False
    )-> None:
        
        super().__init__(parameters, lr)

        assert momentum >= 0.0
        assert dampening >= 0.0
        assert weight_decay >= 0.0

        if nesterov:
            assert momentum > 0.0
        
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        self._velocity: dict[int, torch.Tensor] = {}


    def step(self) -> None:
        """
        Update model parameters.
        """

        with torch.no_grad():

            for parameter in self.parameters:

                if parameter.data.grad is None:
                    continue

                grad = parameter.data.grad

                if self.weight_decay != 0.0:
                    grad = grad + self.weight_decay * parameter.data

                if self.momentum != 0.0:

                    key = id(parameter)

                    if key not in self._velocity:
                        self._velocity[key] = torch.zeros_like(parameter.data)

                    velocity = self._velocity[key]

                    velocity.mul_(self.momentum)
                    velocity.add_(grad, alpha=1.0 - self.dampening)

                    if self.nesterov:
                        grad = grad + self.momentum * velocity
                    else:
                        grad = velocity

                parameter.data.add_(grad, alpha=-self.lr)