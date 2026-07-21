"""
Tests for Hessian computation.
"""

from __future__ import annotations

import torch

from solarpinn.physics.hessian import hessian

def test_quadratic_1d()-> None:
    x = torch.tensor([3.0], requires_grad = True)

    y = x.pow(2)

    H = hessian(y, x)

    expected = torch.tensor([[2.0]])

    assert torch.allclose(H, expected)

def test_cubic_1d()-> None:
    x = torch.tensor([2.0], requires_grad = True)

    y = x.pow(3)

    H = hessian(y, x)

    expected = torch.tensor([[12.0]])

    assert torch.allclose(H, expected)

def test_separable_2d()-> None:
    x = torch.tensor(
            [2.0, 3.0],
            requires_grad = True
            )
    y = (x ** 2).sum()

    H = hessian(y, x)

    expected = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 2.0]
                ]
            )

    assert torch.allclose(H, expected)


def test_mixxed_partials()-> None:
    x = torch.tensor(
            [2.0, 3.0],
            requires_grad = True
            )
    y = x[0] * x[1]

    H = hessian(y, x)

    expected = torch.tensor(
           [ [0.0, 1.0],
            [1.0, 0.0]
            ]
            )


def test_hessian_symmetry()-> None:
    x = torch.tensor(
            [1.5, -0.5],
            requires_grad = True
            )
    y = x[0] ** 2 + x[0] * x[1] + x[1] ** 2

    H = hessian(y, x)

    assert torch.allclose(H, H.T)


def test_hessian_shape() -> None:
    x = torch.randn(5, requires_grad= True)

    y = ( x ** 2).sum()

    H = hessian( y, x)
    
    assert H.shape == (5,5)

    
