"""
Automatic differentiation utilities.
"""

from __future__ import annotations

import torch


def gradient(
    output: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """
    Compute the gradient of a scalar output with respect to the inputs.

    Parameters
    ----------
    output:
        Scalar tensor.

    inputs:
        Tensor with respect to which the gradient is computed.

    create_graph:
        Whether to construct the graph for higher-order derivatives.

    retain_graph:
        Whether to retain the computational graph.

    Returns
    -------
    torch.Tensor
        Gradient having the same shape as ``inputs``.
    """

    assert output.numel() == 1, (
        "gradient() expects a scalar output."
    )

    grad = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]

    assert grad is not None

    return grad


