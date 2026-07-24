"""
Base class for all PDEs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from solarpinn.nn.module import Module


class PDE(ABC):
    """
    Base class for all PDEs.
    """

    @abstractmethod
    def residual(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        """
        Compute the PDE residual.
        """
        raise NotImplementedError

    def __call__(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        return self.residual(model, inputs)

    def __repr__(self)-> str:
        return self.__class__.__name__


