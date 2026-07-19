"""
Terminal spinner.
"""

from __future__ import annotations

import itertools
import sys

from .base import ProgressBase
from .format import format_time


class Spinner(ProgressBase):
    """
    Spinner for tasks with unknown duration.
    """

    FRAMES = (
        "|",
        "/",
        "-",
        "\\",
    )

    def __init__(
        self,
        description: str = "",
    ) -> None:

        super().__init__(description)

        self._frames = itertools.cycle(self.FRAMES)

    def update(self) -> None:

        frame = next(self._frames)

        sys.stdout.write(
            "\r"
            f"{frame} "
            f"{self.description} "
            f"Elapsed: {format_time(self.elapsed)}"
        )

        sys.stdout.flush()

    def finish(self) -> None:

        sys.stdout.write(
            "\r"
            f"{self.description} "
            f"completed in {format_time(self.elapsed)}\n"
        )

        sys.stdout.flush()