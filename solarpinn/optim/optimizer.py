"""
Base class for all optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from solarpinn.nn.parameter import Parameter


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """

    def __init__(
            self,
            parameters: Iterable[Parameter],
            lr: float
    )-> None:
        
        assert lr > 0
        self.parameters = list(parameters)

        assert len(self.parameters) > 0

        self.lr = lr
    
    def zero_grad(self)-> None:
        """
        Clear gradients of all parameters.
        """

        for parameter in self.parameters:
            if parameter.data.grad is not None:
                parameter.data.grad.zero_()

    
    @abstractmethod
    def step(self)-> None:
        """
        Update model parameters.
        """

        raise NotImplementedError
    
