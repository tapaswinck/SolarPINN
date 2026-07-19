"""
Formatting utilities for progress indicators.
"""

from __future__ import annotations

from datetime import timedelta


def format_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.
    """

    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def format_percentage(value: float) -> str:
    """
    Format percentage.
    """

    return f"{value:.1f}%"


def format_fraction(current: int, total: int) -> str:
    """
    Format progress fraction.
    """

    return f"{current}/{total}"


def format_rate(rate: float) -> str:
    """
    Format iterations per second.
    """

    return f"{rate:.2f} it/s"