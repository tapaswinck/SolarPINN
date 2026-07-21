"""
Tests for divergence computation.
"""
from __future__ import annotations

import pytest
import torch

from solarpinn.physics.divergence import divergence

def test_identity() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    F = x

    div = divergence(F, x)

    assert torch.allclose(
        div,
        torch.tensor(2.0),
    )

def test_quadratic() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    F = x ** 2

    div = divergence(F, x)

    assert torch.allclose(
        div,
        torch.tensor(10.0),
    )

def test_mixed() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    F = torch.stack(
        [
            x[0] * x[1],
            x[0] + x[1],
        ]
    )

    div = divergence(F, x)

    assert torch.allclose(
        div,
        torch.tensor(4.0),
    )

def test_constant_field() -> None:

    x = torch.tensor(
        [2.0, 3.0],
        requires_grad=True,
    )

    F = torch.stack(
        [
            x[0] * 0.0,
            x[1] * 0.0,
        ]
    )

    div = divergence(F, x)

    assert torch.allclose(
        div,
        torch.tensor(0.0),
    )

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
        divergence(F, x)

def test_scalar_output() -> None:

    x = torch.randn(
        4,
        requires_grad=True,
    )

    F = x ** 2

    div = divergence(F, x)

    assert div.ndim == 0


