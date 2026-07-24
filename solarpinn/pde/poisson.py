"""
Poisson equation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from solarpinn.nn.module import Module
from solarpinn.physics.laplacian import laplacian

from .equation import PDE

class PoissonEquation(PDE):
    """
    Poisson equation.
    """

    def __init__(
            self,
            source: Callable[[torch.Tensor], torch.Tensor]
            )-> None:
        self.source = source

    def residual(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        prediction = model(inputs)
        
        lap = laplacian(prediction, inputs)

        rhs = self.source(inputs)

        return lap - rhs


