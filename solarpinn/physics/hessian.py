"""
Hessian computation utilities.
"""

from __future__ import annotations

import torch

from .gradients import gradient
from .jacobian import jacobian


def hessian(
        output: torch.Tensor,
        inputs: torch.Tensor
        )-> torch.Tensor:
    """
    Compute the Hessian matrix of a scalar output with respect to the inputs.

    Parameters
    ----------
    output:
        Scalar tensor.

    inputs:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Hessian matrix of shape (inputs.numel(), inputs.numel()).
    """

    assert output.numel() == 1,(
            "hessian expects a scalar output."
            )
    grad = gradient(output, inputs)

    return jacobian(grad, inputs)
