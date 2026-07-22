"""
Curl computation utilities.
"""

from __future__ import annotations

import torch

from .jacobian import jacobian


def curl(
        vector: torch.Tensor,
        inputs: torch.Tensor
        )-> torch.Tensor:

    """
    Compute the curl of a vector field.

    Supports 2D and 3D vector fields.

    Parameters
    ----------
    vector:
        Vector field.

    inputs:
        Coordinates

    Returns
    -------
    torch.Tensor
        Scalar (2D) or vector (3D) curl.
    """
    
    assert vector.ndim ==1,(
            "vector must be one-dimensional."
            )

    assert inputs.ndim == 1, (
            "inputs must be one-dimensional."
            )

    assert vector.numel() == inputs.numel(), (
            "vector and inputs must have the same dimension"
            )

    J = jacobian(vector, inputs)

    if vector.numel() == 2:
        return J[1,0] - J[0,1]

    if vector.numel() == 3:
        return torch.stack(
                (
                    J[2,1] - J[1,2],
                    J[0,2] - J[2,0],
                    J[1,0] - J[0,1]
                    )
                )

    raise ValueError(
            "cural only supports 2D and 3D vector fields."
            )



