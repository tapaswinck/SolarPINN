"""
Tests for Jacobian computation.
"""

from __future__ import annotations

import torch

from solarpinn.physics.jacobian import jacobian


def test_identity()-> None:
    x = torch.tensor(
            [2.0,3.0],
            requires_grad = True
            )
    y = x
    J = jacobian(y, x)

    expected = torch.eye(2)

    assert torch.allclose(J, expected)

def test_squares()-> None:
    x = torch.tensor(
            [2.0, 3.0],
            requires_grad = True
            )

    y = x ** 2
    J = jacobian(y, x)

    expected = torch.tensor(
           [
               [4.0, 0.0],
               [0.0,6.0]
            ]
           )
    assert torch.allclose(J, expected)

def test_mixed_function()-> None:
    x = torch.tensor(
            [2.0, 3.0],
            requires_grad = True
            )
    y = torch.stack(
            [
                x[0] * x[1],
                x[0] + x[1]
                ]
            )
    J = jacobian(y, x)

    expected = torch.tensor(
            [
                [3.0, 2.0],
                [1.0, 1.0]
                ]
            )


    assert torch.allclose(J, expected)



def test_shape()-> None:
    x = torch.randn(
            5,
            requires_grad = True
            )
    y = x ** 2
    J = jacobian(y, x)

    assert J.shape == (5,5)

def test_single_output()-> None:
    
    x = torch.tensor(
            [2.0, 3.0],
            requires_grad = True
            )
    y = torch.stack(
            [
                (x ** 2).sum()
                ]
            )
    J = jacobian(y, x)

    expected = torch.tensor(
            [
                [4.0, 6.0]
                ]
            )
    assert torch.allclose(J, expected)
