"""
Tests for the base optimizer.
"""

from __future__ import annotations

import pytest
import torch

from solarpinn.nn.parameter import Parameter
from solarpinn.optim.optimizer import Optimizer


class DummyOptimizer(Optimizer):

    def step(self) -> None:
        pass


def test_constructor() -> None:

    p = Parameter(torch.ones(3))

    opt = DummyOptimizer([p], lr=0.1)

    assert len(opt.parameters) == 1
    assert opt.lr == 0.1


def test_invalid_lr() -> None:

    p = Parameter(torch.ones(3))

    with pytest.raises(AssertionError):
        DummyOptimizer([p], lr=0)


def test_empty_parameter_list() -> None:

    with pytest.raises(AssertionError):
        DummyOptimizer([], lr=0.1)


def test_zero_grad() -> None:

    p = Parameter(torch.tensor([1.0], requires_grad=True))

    p.data.grad = torch.tensor([5.0])

    opt = DummyOptimizer([p], lr=0.1)

    opt.zero_grad()

    assert torch.equal(
        p.data.grad,
        torch.zeros_like(p.data.grad),
    )