"""
Timing utilities for progress indicators.
"""

from __future__ import annotations

import time


class ProgressTimer:
    """
    High-resolution timer.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the timer."""
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.perf_counter() - self._start
    
    