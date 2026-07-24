"""
Base loss class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

class Loss(ABC):
    """
    Base class for all loss functions.
    """

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
            )-> torch.Tensor:
        """
        Compute the loss
        """

        raise NotImplementedError


    def __call__(
            self,
            *args,
            **kwargs
            )-> torch.Tensor:
        """
        Evaluate the loss.
        """

        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:

        return self.__class__.__name__


