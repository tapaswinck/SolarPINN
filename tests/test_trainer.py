"""
Tests for the generic Trainer.
"""

from __future__ import annotations

import torch
import pytest

from solarpinn.nn.module import Module
from solarpinn.trainer import Trainer
from solarpinn.nn.parameter import Parameter
from solarpinn.optim import SGD

class SimpleModel(Module):
    """
    Single-parameter model.
    """

    def __init__(self)-> None:
        super().__init__()

        self.weight = Parameter(torch.tensor(0.0))


    def forward(self) -> torch.Tensor:
        return self.weight.data

def test_constructor() -> None:
    model = SimpleModel()

    optimizer = SGD(model.parameters(), lr = 0.1)

    trainer = Trainer(model=model, optimizer = optimizer)

    assert trainer.model is model
    assert trainer.optimizer is optimizer

def test_train_step_updates_parameter()-> None:
    
    model = SimpleModel()

    optimizer = SGD(model.parameters(), lr = 0.1)

    trainer = Trainer(model = model, optimizer = optimizer)

    def loss_fn() -> torch.Tensor:
        return (model() - 1.0) ** 2

    before = model.weight.detach().clone()

    loss = trainer.train_step(loss_fn)

    after = model.weight.detach().clone()

    assert isinstance(loss, float)
    assert not torch.equal(before, after)


def test_fit_returns_history()-> None:
    model = SimpleModel()

    optimizer = SGD(model.parameters(), lr = 0.1)

    trainer = Trainer(model = model, optimizer = optimizer)

    def loss_fn() -> torch.Tensor:
        return (model() - 1.0) ** 2

    history = trainer.fit(loss_fn = loss_fn, epochs = 5)

    assert len(history) == 5

    assert all(isinstance(loss, float)
               for loss in history
               )

def test_invalid_epochs() -> None:
    model = SimpleModel()

    optimizer = SGD(model.parameters(), lr = 0.1)

    trainer = Trainer(model = model, optimizer = optimizer)

    def loss_fn() -> torch.Tensor:
        return (model() - 1.0) ** 2

    with pytest.raises(ValueError):
        trainer.fit(loss_fn = loss_fn, epochs = 0)

def test_repr()-> None:
    
    model = SimpleModel()

    optimizer = SGD(model.parameters(), lr = 0.1)

    trainer = Trainer(model = model, optimizer = optimizer)

    assert repr(trainer) == (
            "Trainer("
            "model=SimpleModel, "
            "optimizer=SGD)"
            )

