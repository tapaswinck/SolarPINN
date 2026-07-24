"""
Base sampler for PINNs.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch

class Sampler(ABC):
    """
    Base class for all collocation-point samplers.
    """

    def __init__(
            self,
            bounds: list[tuple[float, float]],
            n_points: int
            )-> None:
        if len(bounds) == 0:
            raise ValueError("bounds cannot be empty.")
        
        if n_points <= 0:
            raise ValueError("n_points must be positive.")

        self.bounds = bounds
        self.n_points = n_points
        self.dimension = len(bounds)


    @abstractmethod
    def sample(self)-> torch.Tensor:
        """
        Generate sample points.

        Returns
        ------
        Tensor of shape
            (n_points, dimension)
        """
        raise NotImplementedError

    def __call__(
            self,
            *args,
            **kwargs
            )-> torch.Tensor:
        return self.sample(*args, **kwargs)

    def __repr__(self)-> str:
        return (
                f"{self.__class__.__name__}("
                f"dimension={self.dimension}, "
                f"n_points={self.n_points})"
                )


