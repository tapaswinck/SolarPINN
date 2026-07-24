"""
Generic training loop
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from solarpinn.nn.module import Module
from solarpinn.optim.optimizer import Optimizer

class Trainer:
    """
    Generic neural network trainer.

    Parameters
    ----------
    model:
        Model to optimize

    optimizer:
        solarpinn optimizer
    """

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer
            )-> None:
        self.model = model
        self.optimizer = optimizer

    
    def train_step(
            self,
            loss_fn: Callable[[], torch.Tensor]
            )-> float:
        """
        Perform one optimization step.

        Parameters
        ----------
        loss_fn:
            callable returning a scalar loss tensor.


        Returns
        -------
        float
            Loss value.
        """

        self.optimizer.zero_grad()
        loss = loss_fn()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def fit(
            self,
            loss_fn: Callable[[], torch.Tensor],
            epochs: int
            )-> list[float]:

        """
        Train the model.

        Parameters
        ----------
        loss_fn:
            Callable returning the training loss.

        epochs:
            Number of optimization steps.


        Returns
        -------
        list[float]
            Loss history.
        """

        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        history: list[float] = []

        for _ in range(epochs):
            history.append(self.train_step(loss_fn)
                           )

        return history

    def __repr__(self)-> str:
        return (
                f"{self.__class__.__name__}("
                f"model={self.model.__class__.__name__}, "
                f"optimizer={self.optimizer.__class__.__name__})"
                )
