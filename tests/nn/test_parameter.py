"""
Unit tests for Parameter.
"""

from __future__ import annotations

import torch

from solarpinn.nn import Parameter


def test_parameter() -> None:
    """
    Test Parameter initialization and methods.
    """

    tensor = torch.randn(3, 4)

    parameter = Parameter(tensor)

    # Basic properties
    assert parameter.shape == torch.Size([3, 4])
    assert parameter.ndim == 2
    assert parameter.numel() == 12
    assert parameter.requires_grad is True
    assert parameter.dtype == tensor.dtype
    assert parameter.device == tensor.device

    # Gradient should initially be None
    assert parameter.grad is None

    # Clone
    clone = parameter.clone()

    assert isinstance(clone, Parameter)
    assert clone is not parameter
    assert torch.equal(clone.data, parameter.data)
    assert clone.requires_grad == parameter.requires_grad

    # Detach
    detached = parameter.detach()

    assert isinstance(detached, torch.Tensor)
    assert detached.requires_grad is False


def main() -> None:
    """
    Run all tests.
    """

    test_parameter()

    print("All Parameter tests passed.")


if __name__ == "__main__":
    main()