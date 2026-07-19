"""
Abstract base class for progress indicators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .timer import ProgressTimer


class ProgressBase(ABC):
    """
    Base class for all progress indicators.
    """

    def __init__(self, description: str = "") -> None:
        self.description = description
        self.timer = ProgressTimer()

    @property
    def elapsed(self) -> float:
        return self.timer.elapsed

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the progress indicator."""

    @abstractmethod
    def finish(self) -> None:
        """Finish the progress indicator."""