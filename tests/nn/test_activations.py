import torch

from solarpinn.nn.activations import(
    Identity,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh
)

def test_identity():

    x = torch.randn(5, 3)

    y = Identity()(x)

    assert torch.equal(x, y)

def test_relu():

    x = torch.tensor([-2., -1., 0., 2.])

    y = ReLU()(x)

    expected = torch.tensor([0., 0., 0., 2.])

    assert torch.equal(y, expected)

def test_leaky_relu():

    x = torch.tensor([-2., 2.])

    y = LeakyReLU(0.1)(x)

    expected = torch.tensor([-0.2, 2.0])

    assert torch.allclose(y, expected)

def test_sigmoid():

    x = torch.tensor([0.])

    y = Sigmoid()(x)

    assert torch.allclose(
        y,
        torch.tensor([0.5]),
    )

def test_tanh():

    x = torch.tensor([0.])

    y = Tanh()(x)

    assert torch.allclose(
        y,
        torch.tensor([0.0]),
    )

    