"""
Laplacian computation utilities.
"""

from __future__ import annotations

import torch

from .hessian import hessian

def laplacian(
        output: torch.Tensor,
        inputs: torch.Tensor
        )-> torch.Tensor:

    """
    Compute the Laplacian of a scalar function.

    Parameters
    ----------
    output: 
        Scalar output tensor.

    inputs:
        Input tensor


    Returns
    -------
    torch.Tensor
        Scalar Laplacian.
    """

    H = hessian(output, inputs)
    
    return torch.trace(H)


