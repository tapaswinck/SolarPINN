"""
Jacobian computation utilities.
"""

from __future__ import annotations

import torch

from .gradients import gradient

def jacobian(
        outputs: torch.Tensor,
        inputs: torch.Tensor
        )-> torch.Tensor:

    """
    Compute the Jacobian matrix of 'outputs' with respect to 'inputs'.

    Parameters
    ----------
    outputs:
        One-dimensional output tensor.

    inputs:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Jacobian of shape(outputs.numel(), inputs.numel()).
    """

    assert outputs.ndim ==1, (
            "outputs must be one-dimensional"
            )
    rows = []
    
    for output in outputs:

        rows.append(
                gradient(output, inputs)
                )

    return torch.stack(rows)



