"""
Mean squared error loss.
"""

from __future__ import annotations

import torch

from .loss import Loss

class MSELoss(Loss):
    """
    Mean squared error (MSE) loss.

    Computes
        mean((prediction - target)^2)
    """

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
            )-> torch.Tensor:
        """
        Copmute thet mean squared error.
        """

        return torch.mean((prediction - target) ** 2)



