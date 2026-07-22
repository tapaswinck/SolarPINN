"""
Base class for all partial differentiation equations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from solarpinn.nn.module import Module

class PDE(ABC):
    """
    Abstract base class for all PDEs.

    A PDE defines a residual function that should evaluate to zero 
    when the neural netwoek satisfies the equuation.
    """

    @abstractmethod
    def residual(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        """
        Compute the PDE residual.

        Parameters
        ----------
        model:
            Neural network approximation.

        inputs:
            Collocation points.

        Returns
        -------
        torch.Tensor:
            Residual evaluated at the collocation points.
        """

        raise NotImplementedError
    
    def loss(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        """
        Compute the mean-squared PDE residual.

        Parameters
        ----------
        model:
            Neural network approximation.

        inputs:
            Collocation points

        Returns
        -------
        torch.Tensor:
            Mean squared residual.
        """


        residual = self.residual(model, inputs)

        return torch.mean(residual ** 2)

    
    def __call__(
            self,
            model: Module,
            inputs: torch.Tensor
            )-> torch.Tensor:
        """
        Evaluate the PDE loss.
        """

        return self.loss(model, inputs)

    def __repr__(self)-> str:
        return self.__class__.__name__



