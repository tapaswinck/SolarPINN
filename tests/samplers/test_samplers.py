"""
Tests for base sampler.
"""

from __future__ import annotations

import pytest
import torch

from solarpinn.samplers.sampler import Sampler

class DummySampler(Sampler):
    
    def sample(self)-> torch.Tensor:

        return torch.zeros(
                self.n_points,
                self.dimension
                )

def test_call()-> None:
    sampler = DummySampler(
            bounds = [(0.0, 1.0)],
            n_points = 10
            )

    samples = sampler()

    assert samples.shape == (10, 1)


def test_repr()-> None:

    sampler = DummySampler(
            bounds = [(0.0, 1.0)],
            n_points = 5
            )

    text = repr(sampler)

    assert "DummySampler" in text
    assert "dimension=1" in text
    assert "n_points=5" in text


def test_invalid_bounds()-> None:

    with pytest.raises(ValueError):
        DummySampler([], 10)


def test_invalid_npoints()-> None:

    with pytest.raises(ValueError):
        DummySampler(
                bounds = [(0.0, 1.0)],
                n_points = 0
                )


def test_dimension()-> None:

    sampler = DummySampler(
            bounds = [
                (0.0, 1.0),
                (-1.0, 1.0),
                (2.0, 5.0)
                ],
            n_points = 100
            )
    assert sampler.dimension == 3


