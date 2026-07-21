"""
Tests for Adam optimizer.
"""

from __future__ import annotations

import torch
import pytest

from solarpinn.nn.parameter import Parameter
from solarpinn.optim.adam import Adam

def test_constructor() -> None:

    p = Parameter(torch.ones(3))

    optimizer = Adam([p])

    assert optimizer.lr == 1e-3
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999

def test_invalid_arguments() -> None:

    p = Parameter(torch.ones(3))

    with pytest.raises(AssertionError):
        Adam([p], lr=0)

    with pytest.raises(AssertionError):
        Adam([p], betas=(-0.1, 0.999))

    with pytest.raises(AssertionError):
        Adam([p], betas=(0.9, 1.1))

    with pytest.raises(AssertionError):
        Adam([p], eps=0)

    with pytest.raises(AssertionError):
        Adam([p], weight_decay=-1)

def test_step() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    p.data.grad = torch.tensor([1.0])

    optimizer = Adam([p], lr=0.1)

    optimizer.step()

    assert p.data < 1.0

def test_multiple_parameters() -> None:

    p1 = Parameter(torch.tensor([1.0], requires_grad=True))
    p2 = Parameter(torch.tensor([2.0], requires_grad=True))

    p1.data.grad = torch.tensor([1.0])
    p2.data.grad = torch.tensor([2.0])

    optimizer = Adam([p1, p2])

    optimizer.step()

    assert p1.data < 1.0
    assert p2.data < 2.0

def test_zero_gradient() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    optimizer = Adam([p])

    optimizer.step()

    assert torch.equal(
        p.data,
        torch.tensor([1.0]),
    )

def test_state_created() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    p.data.grad = torch.tensor([1.0])

    optimizer = Adam([p])

    optimizer.step()

    assert len(optimizer._m) == 1
    assert len(optimizer._v) == 1

def test_multiple_steps() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    optimizer = Adam([p], lr=0.1)

    for _ in range(5):
        p.data.grad = torch.tensor([1.0])
        optimizer.step()

    assert p.data < 0.6


