"""
Divergence computation utilities.
"""

from __future__ import annotations

import torch

from .jacobian import jacobian


def divergence(
        vector: torch.Tensor,
        inputs: torch.Tensor
        )-> torch.Tensor:

    """
    Compute the divergence of a vector field.

    Parameters
    ----------
    
    vector
        One-dimensional vector field.

    inputs
        Input coordinates.

    Returns
    -------
    torch.Tensor
        Scalar divergence.
    """

    assert vector.ndim == 1, (
            "vector must be one-dimensional."
            )

    assert vector.numel() == inputs.numel(), (
            "vector and inputs must have the same length"
            )
    J = jacobian(vector, inputs)

    return torch.trace(J)



