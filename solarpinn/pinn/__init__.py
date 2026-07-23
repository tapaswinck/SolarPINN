"""
Physics-Informed Neural Network components.
"""

from .pde import PDE
from .boundary import (
        BoundaryCondition,
        DirichletBoundary,
        NeumannBoundary,
        RobinBoundary
        )


__all__ = [
        "PDE",
        "BoundaryCondition",
        "DirichletCondiiton",
        "NeumannBoundary",
        "RobinBoundary"
        ]


