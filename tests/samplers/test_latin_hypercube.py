"""
Tests for LatinHypercubeSampler.
"""

from __future__ import annotations

import torch

from solarpinn.samplers.latin_hypercube import(
        LatinHypercubeSampler,
        )

def test_shape()-> None:
    sampler = LatinHypercubeSampler(
            bounds = [
                (0.0, 1.0),
                (-1.0, 1.0)
                ],
            n_points = 100
            )

    samples = sampler()

    assert samples.shape == (100,2)


def test_bounds()-> None:
    sampler = LatinHypercubeSampler(
            bounds = [
                (-2.0, 3.0),
                (5.0, 10.0)
                ],
            n_points = 500
            )

    samples = sampler()

    assert torch.all(samples[:, 0] >= -2.0)
    assert torch.all(samples[:, 0] <= 3.0)
    assert torch.all(samples[:, 1] >= 5.0)
    assert torch.all(samples[:, 1] <= 10.0)


def test_randomness()-> None:
    sampler = LatinHypercubeSampler(
            bounds = [(0.0, 1.0)],
            n_points = 50
            )


    a = sampler()
    b = sampler()

    assert not torch.allclose(a, b)

def test_dimension_one()-> None:
    sampler = LatinHypercubeSampler(
            bounds = [(-1.0, 1.0)],
            n_points = 20
            )

    samples = sampler()

    assert samples.shape == (20, 1)


def test_large_sample() -> None:

    sampler = LatinHypercubeSampler(
            bounds = [
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0)
                ],
            n_points = 5000
            )

    samples = sampler()

    assert samples.shape == (5000, 3)


def test_unique_intervals()-> None:
    """
    Every interval should contain exactly one sample.
    """
    
    n = 100

    sampler = LatinHypercubeSampler(
            bounds = [(0.0, 1.0)],
            n_points = n
            )

    samples = sampler().squeeze()

    bins = torch.floor(samples * n).long()
    bins = torch.sort(bins).values

    assert torch.equal(bins, torch.arange(n))


def test_repr()-> None:

    sampler = LatinHypercubeSampler(
            bounds = [
                (0.0, 1.0),
                (0.0,2.0)
                ],
            n_points = 100

            )

    assert repr(sampler) == (
            "LatinHypercubeSampler("
            "dimension=2, "
            "n_points=100)"
            )


