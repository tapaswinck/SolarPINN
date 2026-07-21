"""
Optimization algorithms.
"""

from solarpinn.optim.optimizer import Optimizer
from solarpinn.optim.sgd import SGD

__all__ = [
    "Optimizer",
    "SGD",
]