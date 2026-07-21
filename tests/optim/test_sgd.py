"""
Unit tests for SGD optimizer.
"""

from __future__ import annotations

import torch

from solarpinn.nn.parameter import Parameter
from solarpinn.optim.sgd import SGD

def test_constructor() -> None:

    p = Parameter(torch.ones(3))

    optimizer = SGD([p], lr=0.1)

    assert optimizer.lr == 0.1
    assert optimizer.momentum == 0.0

def test_step() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    p.data.grad = torch.tensor([2.0])

    optimizer = SGD([p], lr=0.1)

    optimizer.step()

    assert torch.allclose(
        p.data,
        torch.tensor([0.8]),
    )

def test_zero_gradient() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    optimizer = SGD([p], lr=0.1)

    optimizer.step()

    assert torch.equal(
        p.data,
        torch.tensor([1.0]),
    )

def test_multiple_parameters() -> None:

    p1 = Parameter(torch.tensor([1.0], requires_grad=True))
    p2 = Parameter(torch.tensor([2.0], requires_grad=True))

    p1.data.grad = torch.tensor([1.0])
    p2.data.grad = torch.tensor([2.0])

    optimizer = SGD([p1, p2], lr=0.1)

    optimizer.step()

    assert torch.allclose(p1.data, torch.tensor([0.9]))
    assert torch.allclose(p2.data, torch.tensor([1.8]))

def test_momentum() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    optimizer = SGD(
        [p],
        lr=0.1,
        momentum=0.9,
    )

    p.data.grad = torch.tensor([1.0])
    optimizer.step()

    first = p.data.clone()

    p.data.grad = torch.tensor([1.0])
    optimizer.step()

    second = p.data.clone()

    assert second < first

def test_weight_decay() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    p.data.grad = torch.zeros_like(p.data)

    optimizer = SGD(
        [p],
        lr=0.1,
        weight_decay=0.1,
    )

    optimizer.step()

    assert p.data < 1.0

