"""
Loss functions for neural networks.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from .module import Module

class Loss(Module):
    """
    Abstract base class for all loss functions.
    """

    @abstractmethod
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    )-> torch.Tensor:
        """
        Compute the loss
        """

        raise NotImplementedError
    

class MSELoss(Loss):
    """
    Mean Squared Error (MSE).

        L = mean((prediction - target) ** 2)
    """

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    )-> torch.Tensor:
        
        assert prediction.shape == target.shape

        error = prediction - target
        
        return torch.mean(error * error)
    
class L1Loss(Loss):
    """
    Mean Absolute Error (MAE).

        L = mean(|prediction - target|)
    """

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    )-> torch.Tensor:
        
        assert prediction.shape == target.shape

        return torch.mean(torch.abs(prediction - target))
    

class SmoothL1Loss(Loss):
    """
    Smooth L1 Loss.
    
    Equivalent to Huber loss with delta = beta.
    """

    def __init__(
            self,
            beta: float = 1.0
    )-> None:
        super().__init__()

        assert beta >0
        self.beta = beta

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    )-> torch.Tensor:
        
        assert prediction.shape == target.shape

        error = prediction - target
        absolute = torch.abs(error)

        quadratic = 0.5 * error.pow(2) / self.beta
        linear = absolute - 0.5 * self.beta

        return torch.mean(
            torch.where(
                absolute < self.beta,
                quadratic,
                linear
            )
        )
    

class HuberLoss(Loss):
    """
    Huber loss.

    Quadratic for small errors and linear for large errors.
    """

    def __init__(
            self,
            delta: float = 1.0
    )-> None:
        
        super().__init__()

        assert delta > 0

        self.delta = delta


    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    )-> torch.Tensor:
        
        assert prediction.shape == target.shape

        error = prediction - target
        absolute = torch.abs(error)

        quadratic = 0.5 * error.pow(2)
        linear = self.delta * (absolute - 0.5 * self.delta)

        return torch.mean(torch.where(absolute <= self.delta, quadratic, linear))
 

class RMSELoss(Loss):
    """
    Root Mean Squared Error.
    """

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        assert prediction.shape == target.shape

        mse = torch.mean((prediction - target).pow(2))

        return torch.sqrt(mse)
    
class RelativeL2Loss(Loss):
    """
    Relative L2 error.
    """

    def __init__(
            self,
            eps: float = 1e-8
    )-> None:
        
        super().__init__()

        self.eps = eps

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor
    )-> torch.Tensor:
        
        assert prediction.shape == target.shape

        numerator = torch.linalg.norm(prediction - target)
        denominator = torch.linalg.norm(target)

        return numerator / (denominator + self.eps)
    
