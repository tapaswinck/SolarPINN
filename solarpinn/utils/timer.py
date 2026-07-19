"""
Timing utilities.
"""

from __future__ import annotations

import time

from solarpinn.utils.logger import get_logger

class Timer:
    """
    Context manager for timing code execution.

    Parameters
    ----------
    name: str
        Name of the timed block.
    
    logger_name: str
        Logger name.
    """
    def __init__(
            self,
            name: str = "Execution",
            logger_name: str = __name__
    ) -> None:
        self.name = name
        self.logger = get_logger(logger_name)

    def __enter__(self)-> "Timer":
        self.start = time.perf_counter()
        return self
    
    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:

        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

        if exc_type is None:
            self.logger.info(
                "%s completed in %.3f s.",
                self.name,
                self.elapsed,
            )
        else:
            self.logger.exception(
                "%s failed after %.3f s.",
                self.name,
                self.elapsed,
            )

        return False

        