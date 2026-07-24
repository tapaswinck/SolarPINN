"""
Uniform random sampler.
"""

from __future__ import annotations

import torch

from .sampler import Sampler

class UniformSampler(Sampler):
    """
    Uniform random collocation-point sampler.
    """

    def sample(self)-> torch.Tensor:
        """
        Sample uniformly inside the domain.

        Returns
        -------
        Tensor of shape
            (n_points, dimension)
        """

        samples = torch.empty(
                self.n_points,
                self.dimension)

        for i, (low, high) in enumerate(self.bounds):
            samples[:, i] = torch.rand(self.n_points) * (high - low) + low

        return samples

    
