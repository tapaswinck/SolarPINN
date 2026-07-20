import torch
import math

from solarpinn.nn.initialization import (
    zeros,
    ones,
    uniform,
    normal,
    calculate_fan_in_and_fan_out,
    xavier_normal,
    xavier_uniform,
    kaiming_normal,
    kaiming_uniform
)


def test_zeros():

    x = torch.empty(4, 3)

    zeros(x)

    assert torch.all(x == 0)


def test_ones():

    x = torch.empty(4, 3)

    ones(x)

    assert torch.all(x == 1)


def test_uniform():

    x = torch.empty(4, 3)

    uniform(x)

    assert x.shape == (4, 3)


def test_normal():

    x = torch.empty(4, 3)

    normal(x)

    assert x.shape == (4, 3)


def test_fan():

    x = torch.empty(64, 32)

    fan_in, fan_out = calculate_fan_in_and_fan_out(x)

    assert fan_in == 32
    assert fan_out == 64


x = torch.empty(128, 64)

xavier_uniform(x)

fan_in, fan_out = calculate_fan_in_and_fan_out(x)

bound = math.sqrt(6.0 / (fan_in + fan_out))

assert torch.all(x <= bound)
assert torch.all(x >= -bound)



x = torch.empty(10000, 64)

xavier_normal(x)

assert abs(x.mean().item()) < 0.05


x = torch.empty(128, 64)

kaiming_uniform(x)

assert x.shape == (128, 64)


x = torch.empty(10000, 64)

kaiming_normal(x)

assert abs(x.mean().item()) < 0.05


def main():

    test_zeros()
    test_ones()
    test_uniform()
    test_normal()
    test_fan()

    print("All initialization tests passed.")


if __name__ == "__main__":
    main()


