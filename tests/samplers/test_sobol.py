"""
Tests for SobolSampler.
"""

from __future__ import annotations

import torch

from solarpinn.samplers.sobol import SobolSampler


def test_shape() -> None:

    sampler = SobolSampler(
        bounds=[
            (0.0, 1.0),
            (-1.0, 1.0),
        ],
        n_points=128,
    )

    samples = sampler()

    assert samples.shape == (128, 2)


def test_bounds() -> None:

    sampler = SobolSampler(
        bounds=[
            (-2.0, 3.0),
            (5.0, 10.0),
        ],
        n_points=256,
    )

    samples = sampler()

    assert torch.all(samples[:, 0] >= -2.0)
    assert torch.all(samples[:, 0] <= 3.0)

    assert torch.all(samples[:, 1] >= 5.0)
    assert torch.all(samples[:, 1] <= 10.0)


def test_dimension_one() -> None:

    sampler = SobolSampler(
        bounds=[(-1.0, 1.0)],
        n_points=64,
    )

    samples = sampler()

    assert samples.shape == (64, 1)


def test_large_sample() -> None:

    sampler = SobolSampler(
        bounds=[
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ],
        n_points=4096,
    )

    samples = sampler()

    assert samples.shape == (4096, 3)


def test_scrambled_randomness() -> None:

    sampler = SobolSampler(
        bounds=[(0.0, 1.0)],
        n_points=128,
        scramble=True,
    )

    a = sampler()
    b = sampler()

    assert not torch.allclose(a, b)


def test_unscrambled_deterministic() -> None:

    sampler = SobolSampler(
        bounds=[(0.0, 1.0)],
        n_points=128,
        scramble=False,
    )

    a = sampler()
    b = sampler()

    assert torch.allclose(a, b)


def test_repr() -> None:

    sampler = SobolSampler(
        bounds=[
            (0.0, 1.0),
            (0.0, 2.0),
        ],
        n_points=100,
    )

    assert repr(sampler) == (
        "SobolSampler("
        "dimension=2, "
        "n_points=100, "
        "scramble=True)"
    )
