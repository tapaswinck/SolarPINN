"""
Tests for gradient computation.
"""

from __future__ import annotations

import torch

from solarpinn.physics import gradient

def test_gradient_square() -> None:

    x = torch.tensor([2.0], requires_grad=True)

    y = x.pow(2)

    grad = gradient(y, x)

    assert torch.allclose(
        grad,
        torch.tensor([4.0]),
    )

def test_gradient_cube() -> None:

    x = torch.tensor([3.0], requires_grad=True)

    y = x.pow(3)

    grad = gradient(y, x)

    assert torch.allclose(
        grad,
        torch.tensor([27.0]),
    )

def test_gradient_vector() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    y = (x ** 2).sum()

    grad = gradient(y, x)

    expected = torch.tensor([4.0, 6.0])

    assert torch.allclose(
        grad,
        expected,
    )

def test_second_derivative() -> None:

    x = torch.tensor([2.0], requires_grad=True)

    y = x.pow(3)

    first = gradient(y, x)

    second = gradient(first.sum(), x)

    assert torch.allclose(
        second,
        torch.tensor([12.0]),
    )

def test_gradient_shape() -> None:

    x = torch.randn(
        5,
        requires_grad=True,
    )

    y = (x ** 2).sum()

    grad = gradient(y, x)

    assert grad.shape == x.shape


