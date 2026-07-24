"""
Latin Hypercube sampler.
"""

from __future__ import annotations

import torch

from .sampler import Sampler


class LatinHypercubeSampler(Sampler):
    """
    Latin Hypercube Sampling (LHS).

    Generates one sample from eeach interval in every dimension, 
    then randomly permutes the intervals independently.
    """
    
    def sample(self)-> torch.Tensor:
        """
        Generate Latin Hypercube samples.

        Returns
        -------
        Tensor of shape
            (n_points, dimension)
        """

        samples = torch.empty(
                self.n_points,
                self.dimension
                )

        n = self.n_points

        for d, (low, high) in enumerate(self.bounds):
            #Random point inside each interval.
            u = torch.rand(n)

            intervals = (torch.arange(n) + u) / n

            #Shuffle independently
            permutation = torch.randperm(n)

            intervals = intervals[permutation]

            #Scale to domain
            samples[:, d] = low + intervals *(high - low)


        return samples



