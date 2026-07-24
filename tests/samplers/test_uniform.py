"""
Tests for UniformSampler.
"""

from __future__ import annotations

import torch

from solarpinn.samplers.uniform import UniformSampler


def test_shape()-> None:

    sampler = UniformSampler(
            bounds = [

                (0.0, 1.0),
                (-2.0, 2.0)
                ],
            n_points = 500
            )
    samples = sampler()

    assert samples.shape == (500, 2)


def test_bounds()-> None:
    sampler = UniformSampler(
            bounds = [
                (-1.0, 1.0),
                (5.0, 10.0)
                ],
            n_points = 1000
            )

    samples = sampler()

    assert torch.all(samples[:, 0] >= -1.0)
    assert torch.all(samples[:, 0] <= 1.0)

    assert torch.all(samples[:, 1] >= 5.0)
    assert torch.all(samples[:, 1] <= 10.0)

def test_randomness()-> None:
    sampler = UniformSampler(
            bounds = [(0.0,1.0)],
            n_points = 100
            )
    a = sampler()
    b = sampler()
    assert not torch.allclose(a, b)


def test_dimension_one()-> None:

    sampler = UniformSampler(
            bounds = [(-3.0, 3.0)],
            n_points = 20
            )

    samples = sampler()

    assert samples.shape == (20, 1)


def test_large_sample()-> None:
    sampler = UniformSampler(
            bounds = [(0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0)
                      ],
            n_points = 10000
            )

    samples = sampler()

    assert samples.shape == (10000, 3)


