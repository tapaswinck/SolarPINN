"""
Terminal progress bar.
"""

from __future__ import annotations

import sys

from .base import ProgressBase
from .format import (
    format_fraction,
    format_percentage,
    format_rate,
    format_time,
)


class ProgressBar(ProgressBase):
    """
    Terminal progress bar.
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        width: int = 40,
        fill: str = "=",
        empty: str = "-",
    ) -> None:

        super().__init__(description)

        if total <= 0:
            raise ValueError("total must be positive.")

        self.total = total
        self.current = 0

        self.width = width
        self.fill = fill
        self.empty = empty

    @property
    def percentage(self) -> float:
        return 100.0 * self.current / self.total

    @property
    def rate(self) -> float:
        if self.elapsed == 0:
            return 0.0

        return self.current / self.elapsed

    @property
    def eta(self) -> float:
        if self.rate == 0:
            return 0.0

        return (self.total - self.current) / self.rate

    def update(self, step: int = 1) -> None:

        self.current = min(self.current + step, self.total)

        filled = int(self.width * self.current / self.total)

        bar = (
            self.fill * filled
            + self.empty * (self.width - filled)
        )

        sys.stdout.write(
            "\r"
            f"{self.description:<20}"
            f"[{bar}] "
            f"{format_percentage(self.percentage)} "
            f"{format_fraction(self.current, self.total)} "
            f"{format_rate(self.rate)} "
            f"ETA {format_time(self.eta)}"
        )

        sys.stdout.flush()

    def finish(self) -> None:

        self.current = self.total
        self.update(step=0)

        sys.stdout.write("\n")
        sys.stdout.flush()