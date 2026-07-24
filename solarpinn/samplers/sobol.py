"""
Sobol sequence sampler.
"""

from __future__ import annotations

import torch

from .sampler import Sampler


class SobolSampler(Sampler):
    """
    Sobol low-discrepancy sampler.

    Generates quasi-random points that cover the domain
    more uniformly than purely random sampling.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        n_points: int,
        scramble: bool = True,
    ) -> None:
        super().__init__(bounds, n_points)

        self.scramble = scramble

    def sample(self) -> torch.Tensor:
        """
        Generate Sobol samples.

        Returns
        -------
        Tensor of shape
            (n_points, dimension)
        """

        engine = torch.quasirandom.SobolEngine(
            dimension=self.dimension,
            scramble=self.scramble,
        )

        samples = engine.draw(self.n_points)

        for d, (low, high) in enumerate(self.bounds):
            samples[:, d] = low + (high - low) * samples[:, d]

        return samples

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dimension={self.dimension}, "
            f"n_points={self.n_points}, "
            f"scramble={self.scramble})"
        )
