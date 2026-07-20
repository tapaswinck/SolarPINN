"""
Base class for all neural network modules.
"""

from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Iterator

from .parameter import Parameter

class Module(ABC):
    """
    Base class for all neural network modules.
    """
    _parameters: dict[str, Parameter]
    _modules: dict[str, "Module"]

    def __init__(self)-> None:
        object.__setattr__(self, "_parameters", dict[str, Parameter]())
        object.__setattr__(self, "_modules", dict[str, Module]())

        self.training = True

    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Automatically register parameters and child modules.
        """

        if isinstance(value, Parameter):
            self._parameters[name] = value

        elif isinstance(value, Module):
            self._modules[name] = value

        object.__setattr__(self, name, value)

    def parameters(self)-> Iterator[Parameter]:
        """
        Iterate over all parameters recursively.
        """

        yield from self._parameters.values()

        for module in self._modules.values():
            yield from module.parameters()

    def children(self)-> Iterator["Module"]:
        """
        Iterate over immediate child modules.
        """

        yield from self._modules.values()

    def modules(self)-> Iterator["Module"]:
        """
        Iterate over this module and all descendants.
        """

        yield self
        
        for module in self._modules.values():
            yield from module.modules()


    def train(self)-> None:
        """
        Set this module and all children to training mode.
        """

        self.training = True

        for module in self.children():
            module.train()

    def eval(self)-> None:
        """
        Set this module and all children to evaluation mode.
        """

        self.training = False

        for module in self.children():
            module.eval()

    
    def __call__(self, *args, **kwargs):
        """
        Invoke the forward pass.
        """

        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Compute the forward pass.
        """

        raise NotImplementedError
    
    def __repr__(self)-> str:
        return self.__class__.__name__
    
    