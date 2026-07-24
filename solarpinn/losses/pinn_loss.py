"""
Composite loss for Physics-Informed Neural Networks.
"""

from __future__ import annotations

import torch

from solarpinn.nn.module import Module

from .loss import Loss

class PINNLoss(Loss):
    """
    Composite PINN loss.

    Sums multiple loss terms such s PDE, boundary,
    initial, or regularization losses.
    """

    def __init__(self)-> None:
        self._terms: list = []

    def add(self, term)-> None:
        """
        Register a new loss term.
        """

        self._terms.append(term)

    def forward(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        """
        Compute the total PINN loss.
        """

        total = torch.tensor(0.0, device = inputs.device,dtype = inputs.dtype)

        for term in self._terms:
            total = total + term.loss(model, inputs)


        return total

    def __len__(self)-> int:
        return len(self._terms)

    def __repr__(self)-> str:
        return f"PINNLoss(n_terms={len(self)})"


