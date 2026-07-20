"""
Trainable parameter implementation.

This module defines the Parameter class, which waps a torch.Tensor 
and represents a trainable parameter within a neural network module.
"""

from __future__ import annotations

from typing import Any

import torch

class Parameter:
    """
    A trainable tensor.

    Parameters
    ----------
    data: torch.Tensor
        Tensor containing the parameter values.
    
    requires_grad: bool, default = True
        Whether gradients should be computed.
    """

    def __init__(
            self,
            data: torch.Tensor,
            requires_grad: bool = True
    )-> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        
        self._data = data.detach().clone()
        self._data.requires_grad_(requires_grad)

    @property
    def data(self) -> torch.Tensor:
        """Underlying tensor."""
        return self._data
    
    @property
    def grad(self)-> torch.Tensor | None:
        """Gradient tensor."""
        return self._data.grad
    
    @property
    def requires_grad(self)->bool:
        """Whether gradients are enabled."""
        return self._data.requires_grad
    
    @property
    def shape(self)-> torch.Size:
        """Tensor shape"""
        return self._data.shape
    
    @property
    def dtype(self)-> torch.dtype:
        """Tensor datatype."""
        return self._data.dtype
    
    @property
    def device(self) -> torch.device:
        """Tensor device."""
        return self._data.device
    
    @property
    def ndim(self)-> int:
        """Number of dimensions."""
        return self._data.ndim
    
    def zero_grad(self)-> None:
        """Reset gradients."""

        if self._data.grad is not None:
            self._data.grad.zero_()

    def clone(self)->"Parameter":
        """Clone the parameter."""

        return Parameter(
            self._data.clone(),
            requires_grad=self.requires_grad
        )
    
    def detach(self)-> torch.Tensor:
        """Return a detached tensor."""

        return self._data.detach()
    
    def numel(self)-> int:
        """Number of elements."""

        return self._data.numel()
    
    def __repr__(self)-> str:
        return(
            f"Parameters("
            f"shape={tuple(self.shape)},"
            f"dtype={self.dtype},"
            f"device={self.device},"
            f"requires_grad={self.requires_grad}"
        )
    
    


