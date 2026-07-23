"""
Physics-Informed Neural Network components.
"""

from .pde import PDE
from .boundary import (
        BoundaryCondition,
        DirichletBoundary,
        NeumannBoundary,
        RobinBoundary,
        PeriodicBoundary
        )

from .initial import(
        InitialCondition,
        ValueInitialCondition,
        DerivativeInitialCondition
        )



__all__ = [
        "PDE",
        "BoundaryCondition",
        "DirichletCondiiton",
        "NeumannBoundary",
        "RobinBoundary",
        "PeriodicBoundary",
        "InitialCondition",
        "ValueInitialCondition",
        "DerivativeInitialCondition"

        ]


