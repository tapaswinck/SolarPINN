"""
Tests for curl computation.
"""

from __future__ import annotations

import pytest
import torch

from solarpinn.physics.curl import curl

def test_identity_2d()-> None:
    x = torch.tensor(
            [2.0, 3.0],
            requires_grad = True
            )
    F = x
    result = curl(F, x)

    assert torch.allclose(result, torch.tensor(0.0))

def test_rotation_field() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    F = torch.stack(
        (
            -x[1],
            x[0],
        )
    )

    result = curl(F, x)

    assert torch.allclose(
        result,
        torch.tensor(2.0),
    )

def test_conservative_field() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    F = torch.stack(
        (
            2 * x[0],
            2 * x[1],
        )
    )

    result = curl(F, x)

    assert torch.allclose(
        result,
        torch.tensor(0.0),
    )

def test_identity_3d() -> None:

    x = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    F = x

    result = curl(F, x)

    expected = torch.zeros(3)

    assert torch.allclose(
        result,
        expected,
    )

def test_rotation_3d() -> None:

    x = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    F = torch.stack(
        (
            -x[1],
            x[0],
            torch.zeros_like(x[0]),
        )
    )

    result = curl(F, x)

    expected = torch.tensor(
        [
            0.0,
            0.0,
            2.0,
        ]
    )

    assert torch.allclose(
        result,
        expected,
    )

def test_invalid_dimension() -> None:

    x = torch.randn(
        4,
        requires_grad=True,
    )

    F = x

    with pytest.raises(ValueError):
        curl(F, x)

def test_shape_mismatch() -> None:

    x = torch.randn(
        3,
        requires_grad=True,
    )

    F = torch.randn(
        2,
        requires_grad=True,
    )

    with pytest.raises(AssertionError):
        curl(F, x)


