"""
Unit tests for the Module class.
"""

from __future__ import annotations

import torch

from solarpinn.nn.module import Module
from solarpinn.nn.parameter import Parameter


class Dummy(Module):
    """
    Simple module used for testing.
    """

    def __init__(self) -> None:
        super().__init__()

        self.weight = Parameter(torch.randn(3, 3))
        self.bias = Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Parent(Module):
    """
    Module containing another module.
    """

    def __init__(self) -> None:
        super().__init__()

        self.layer = Dummy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def test_parameter_registration() -> None:
    model = Dummy()

    parameters = list(model.parameters())

    assert len(parameters) == 2
    assert all(isinstance(parameter, Parameter) for parameter in parameters)


def test_module_registration() -> None:
    model = Parent()

    children = list(model.children())

    assert len(children) == 1
    assert isinstance(children[0], Dummy)


def test_recursive_parameters() -> None:
    model = Parent()

    parameters = list(model.parameters())

    assert len(parameters) == 2


def test_modules() -> None:
    model = Parent()

    modules = list(model.modules())

    assert len(modules) == 2
    assert isinstance(modules[0], Parent)
    assert isinstance(modules[1], Dummy)


def test_training_mode() -> None:
    model = Parent()

    assert model.training is True
    assert model.layer.training is True

    model.eval()

    assert model.training is False
    assert model.layer.training is False

    model.train()

    assert model.training is True
    assert model.layer.training is True


def test_call() -> None:
    model = Dummy()

    x = torch.randn(5, 3)

    y = model(x)

    assert torch.equal(x, y)


def main() -> None:
    test_parameter_registration()
    test_module_registration()
    test_recursive_parameters()
    test_modules()
    test_training_mode()
    test_call()

    print("All Module tests passed.")


if __name__ == "__main__":
    main()