"""
Tests for Laplacian computation.
"""

from __future__ import annotations

import torch

from solarpinn.physics.laplacian import laplacian

def test_quadratic()-> None:
    x = torch.tensor([2.0], requires_grad = True)

    y = x.pow(2)

    assert torch.allclose(laplacian(y, x), torch.tensor(2.0))


def test_cubic()-> None:
    x = torch.tensor([2.0], requires_grad = True)

    y = x.pow(3)

    assert torch.allclose(laplacian(y,x), torch.tensor(12.0))


def test_two_variables() -> None:
    x = torch.tensor([2.0, 3.0], requires_grad = True)

    y = ( x**2).sum()

    assert torch.allclose(laplacian(y,x), torch.tensor(4.0))

def test_cross_term() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    y = x[0] * x[1]

    assert torch.allclose(
        laplacian(y, x),
        torch.tensor(0.0),
    )


def test_mixed_polynomial() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    y = x[0] ** 2 + x[0] * x[1] + x[1] ** 2

    assert torch.allclose(
        laplacian(y, x),
        torch.tensor(4.0),
    )

def test_scalar_output() -> None:
    """
    Laplacian should return a scalar tensor.
    """

    x = torch.randn(
        5,
        requires_grad=True,
    )

    y = (x ** 2).sum()

    result = laplacian(y, x)

    assert result.ndim == 0
