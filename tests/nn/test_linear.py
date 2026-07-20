"""
Unit tests for the Linear layer.
"""

from __future__ import annotations

import torch

from solarpinn.nn.linear import Linear
from solarpinn.nn.parameter import Parameter


def test_shapes() -> None:

    layer = Linear(10, 64)

    x = torch.randn(32, 10)

    y = layer(x)

    assert y.shape == (32, 64)


def test_parameters() -> None:

    layer = Linear(5, 3)

    parameters = list(layer.parameters())

    assert len(parameters) == 2

    assert all(
        isinstance(parameter, Parameter)
        for parameter in parameters
    )


def test_no_bias() -> None:

    layer = Linear(
        8,
        4,
        bias=False,
    )

    assert layer.bias is None

    assert len(list(layer.parameters())) == 1


def test_repr() -> None:

    layer = Linear(4, 2)

    text = repr(layer)

    assert "Linear" in text
    assert "in_features=4" in text
    assert "out_features=2" in text


def main():

    test_shapes()
    test_parameters()
    test_no_bias()
    test_repr()

    print("All Linear tests passed.")


if __name__ == "__main__":
    main()