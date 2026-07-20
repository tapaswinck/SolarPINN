"""
Linear layer.
"""

from __future__ import annotations

import torch

from .initialization import xavier_uniform
from .module import Module
from .parameter import Parameter


class Linear(Module):
    """
    Fully connected layer.

    Computes    
        y = x(W^T) + b
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True
    )-> None:
        super().__init__()

        assert in_features > 0
        assert out_features > 0

        self.in_features = in_features
        self.out_features = out_features

        weight  = torch.empty(
            out_features,
            in_features
        )

        xavier_uniform(weight)

        self.weight = Parameter(weight)

        if bias:
            self.bias = Parameter(
                torch.zeros(out_features)
            )
        else:
            self.bias = None


    def forward(
            self,
            x: torch.Tensor
    )-> torch.Tensor:
        """
        Forward pass.
        """

        y = x @ self.weight.data.T

        if self.bias is not None:
            y = y + self.bias.data

        return y
        
    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )