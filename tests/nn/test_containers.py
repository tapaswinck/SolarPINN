"""
Unit tests for neural network containers.
"""

from __future__ import annotations

import torch

from solarpinn.nn.containers import ModuleDict, ModuleList, Sequential
from solarpinn.nn.module import Module


class Dummy(Module):
    """
    Dummy module used for testing.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_module_list() -> None:
    """
    Test ModuleList.
    """

    modules = ModuleList()

    modules.append(Dummy())
    modules.append(Dummy())

    assert len(modules) == 2

    assert isinstance(modules[0], Dummy)
    assert isinstance(modules[1], Dummy)

    children = list(modules.children())

    assert len(children) == 2
    assert all(isinstance(child, Dummy) for child in children)


def test_module_dict() -> None:
    """
    Test ModuleDict.
    """

    modules = ModuleDict()

    modules.add("encoder", Dummy())
    modules.add("decoder", Dummy())

    assert len(modules) == 2

    assert isinstance(modules["encoder"], Dummy)
    assert isinstance(modules["decoder"], Dummy)

    assert "encoder" in modules.keys()
    assert "decoder" in modules.keys()

    children = list(modules.children())

    assert len(children) == 2
    assert all(isinstance(child, Dummy) for child in children)


def test_sequential() -> None:
    """
    Test Sequential.
    """

    model = Sequential(
        Dummy(),
        Dummy(),
        Dummy(),
    )

    x = torch.randn(8, 4)

    y = model(x)

    assert torch.equal(x, y)

    assert len(list(model.children())) == 3


def main() -> None:
    """
    Run all tests.
    """

    test_module_list()
    test_module_dict()
    test_sequential()

    print("All container tests passed.")


if __name__ == "__main__":
    main()