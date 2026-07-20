"""
Module for activation functions.
"""

from .module import Module

import torch

class Identity(Module):
    """
    Identity actiavation.
    """

    def forward(
            self,
            x: torch.Tensor
    )-> torch.Tensor:
        return x
    
class ReLU(Module):
    """
    Rectified Linear Unit.
    """

    def forward(
            self,
            x: torch.Tensor
    )-> torch.Tensor:
        
        return torch.maximum(x, torch.zeros_like(x))
    

class LeakyReLU(Module):
    """
    Leaky ReLU activation.
    """

    def __init__(
        self,
        negative_slope: float = 0.01,
    ) -> None:

        super().__init__()

        self.negative_slope = negative_slope

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.where(
            x > 0,
            x,
            self.negative_slope * x,
        )
    


class Sigmoid(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
class Tanh(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)